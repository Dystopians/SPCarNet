#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.apply_surface_conditioned_residual_unet_checkpoint import _build_model  # noqa: E402
from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import (  # noqa: E402
    build_lpips_model,
    evidence_views,
    image_lpips_chw,
    image_ssim_chw,
)
from scripts.car_model.train_surface_conditioned_residual_unet import (  # noqa: E402
    _append_alpha_channel_chw,
    _load_face_ids_tensor,
    _load_input_chw,
    _predict_delta_tiled,
    _to_chw,
)
from scripts.car_model.train_perceptual_surface_residual_decoder import (  # noqa: E402
    SurfaceResidualDecoder as PerceptualSurfaceResidualDecoder,
    _face_indices as _decoder_face_indices,
    _feature_dim as _decoder_feature_dim,
    _load_feature_rows as _decoder_load_feature_rows,
    _valid_mask as _decoder_valid_mask,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_png_chw(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.moveaxis(arr, -1, 0).astype(np.float32)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    if mask is not None:
        diff = diff[:, mask]
    return float(np.mean(diff * diff)) if diff.size else 0.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = _mse(a, b)
    if mse <= 1.0e-12:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _valid_mask(z: np.lib.npyio.NpzFile, min_alpha: float) -> np.ndarray:
    face_id = np.asarray(z["face_id"], dtype=np.int64)
    mask = face_id >= 0
    if "barycentric_valid" in z:
        mask &= np.asarray(z["barycentric_valid"]).astype(bool)
    if "alpha" in z:
        mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
    return mask


def _residual_stats(pred_delta: np.ndarray, target_delta: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_delta, dtype=np.float32)[:, mask]
    target = np.asarray(target_delta, dtype=np.float32)[:, mask]
    if pred.size == 0 or target.size == 0:
        return {
            "pixel_count": 0,
            "target_energy": 0.0,
            "pred_energy": 0.0,
            "energy_retention": 0.0,
            "residual_mse": 0.0,
            "residual_psnr": 0.0,
            "cosine": 0.0,
            "sign_agreement": 0.0,
            "changed_fraction": 0.0,
        }
    target_energy = float(np.mean(np.sum(target * target, axis=0)))
    pred_energy = float(np.mean(np.sum(pred * pred, axis=0)))
    diff = pred - target
    residual_mse = float(np.mean(diff * diff))
    dot = float(np.sum(pred.astype(np.float64) * target.astype(np.float64)))
    denom = math.sqrt(float(np.sum(pred.astype(np.float64) ** 2)) * float(np.sum(target.astype(np.float64) ** 2)))
    nonzero = (np.abs(pred) > 1.0e-6) & (np.abs(target) > 1.0e-6)
    return {
        "pixel_count": int(mask.sum()),
        "target_energy": target_energy,
        "pred_energy": pred_energy,
        "energy_retention": float(pred_energy / max(target_energy, 1.0e-12)),
        "residual_mse": residual_mse,
        "residual_psnr": float(-10.0 * math.log10(max(residual_mse, 1.0e-12))),
        "cosine": float(dot / denom) if denom > 1.0e-12 else 0.0,
        "sign_agreement": float(np.mean(np.sign(pred[nonzero]) == np.sign(target[nonzero]))) if np.any(nonzero) else 0.0,
        "changed_fraction": float(np.mean(np.any(np.abs(pred) > (0.5 / 255.0), axis=0))),
    }


def _region_rows(pred_delta: np.ndarray, target_delta: np.ndarray, valid: np.ndarray, grid: int) -> list[dict[str, Any]]:
    _, h, w = pred_delta.shape
    rows: list[dict[str, Any]] = []
    grid = max(1, int(grid))
    for gy in range(grid):
        y0 = int(round(gy * h / grid))
        y1 = int(round((gy + 1) * h / grid))
        for gx in range(grid):
            x0 = int(round(gx * w / grid))
            x1 = int(round((gx + 1) * w / grid))
            mask = np.zeros((h, w), dtype=bool)
            mask[y0:y1, x0:x1] = valid[y0:y1, x0:x1]
            stats = _residual_stats(pred_delta, target_delta, mask)
            rows.append({"region": f"{gy}_{gx}", "y0": y0, "y1": y1, "x0": x0, "x1": x1, **stats})
    return rows


def _summarize_residual_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if int(row.get("pixel_count", 0)) > 0]
    return {
        "view_count": int(len(rows)),
        "valid_view_count": int(len(usable)),
        "target_energy": _mean([float(row["target_energy"]) for row in usable]),
        "pred_energy": _mean([float(row["pred_energy"]) for row in usable]),
        "energy_retention": _mean([float(row["energy_retention"]) for row in usable]),
        "energy_retention_quantiles": _quantiles([float(row["energy_retention"]) for row in usable]),
        "residual_mse": _mean([float(row["residual_mse"]) for row in usable]),
        "residual_psnr": _mean([float(row["residual_psnr"]) for row in usable]),
        "cosine": _mean([float(row["cosine"]) for row in usable]),
        "cosine_quantiles": _quantiles([float(row["cosine"]) for row in usable]),
        "sign_agreement": _mean([float(row["sign_agreement"]) for row in usable]),
        "changed_fraction": _mean([float(row["changed_fraction"]) for row in usable]),
    }


def _summarize_image_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if not row.get("missing", False)]
    return {
        "view_count": int(len(rows)),
        "valid_view_count": int(len(usable)),
        "parent_psnr": _mean([float(row["parent_psnr"]) for row in usable]),
        "candidate_psnr": _mean([float(row["candidate_psnr"]) for row in usable]),
        "psnr_gain": _mean([float(row["candidate_psnr"]) - float(row["parent_psnr"]) for row in usable]),
        "parent_ssim": _mean([float(row["parent_ssim"]) for row in usable]),
        "candidate_ssim": _mean([float(row["candidate_ssim"]) for row in usable]),
        "ssim_gain": _mean([float(row["candidate_ssim"]) - float(row["parent_ssim"]) for row in usable]),
        "parent_lpips": _mean([float(row["parent_lpips"]) for row in usable if row.get("parent_lpips") is not None]),
        "candidate_lpips": _mean(
            [float(row["candidate_lpips"]) for row in usable if row.get("candidate_lpips") is not None]
        ),
        "lpips_gain": _mean(
            [
                float(row["parent_lpips"]) - float(row["candidate_lpips"])
                for row in usable
                if row.get("parent_lpips") is not None and row.get("candidate_lpips") is not None
            ]
        ),
    }


def _compact_audit(audit: dict[str, Any], keep_rows: bool, worst_region_count: int) -> dict[str, Any]:
    if keep_rows:
        return audit
    compact = dict(audit)
    rows = list(compact.pop("rows", []) or [])
    region_rows = list(compact.pop("region_rows", []) or [])
    compact["row_count"] = int(len(rows))
    compact["region_row_count"] = int(len(region_rows))
    compact["rows_preview"] = rows[: min(4, len(rows))]
    usable_regions = [row for row in region_rows if int(row.get("pixel_count", 0) or 0) > 0]
    usable_regions.sort(key=lambda row: (float(row.get("cosine", 0.0) or 0.0), -float(row.get("residual_mse", 0.0) or 0.0)))
    compact["worst_regions_by_cosine"] = usable_regions[: max(0, int(worst_region_count))]
    return compact


def _support_summary(model: torch.nn.Module, features: torch.Tensor, face_ids: torch.Tensor | None, valid: np.ndarray) -> dict[str, float]:
    if face_ids is None or not hasattr(model, "support_mask"):
        return {}
    with torch.no_grad():
        support = (
            model.support_mask(features.unsqueeze(0).to(next(model.parameters()).device), face_ids.unsqueeze(0).to(next(model.parameters()).device))
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            > 0.5
        )
    valid_count = max(1, int(np.sum(valid)))
    return {
        "known_face_fraction": float(np.sum((face_ids.numpy() > 0) & valid) / valid_count),
        "active_support_fraction": float(np.sum(support & valid) / valid_count),
    }


def _policy_val_paths(evidence_dir: Path, stride: int, max_views: int) -> list[Path]:
    paths = evidence_views(evidence_dir)
    selected = [path for idx, path in enumerate(paths) if int(stride) > 1 and idx % int(stride) == 0]
    if not selected:
        selected = paths
    if int(max_views) > 0:
        selected = selected[: int(max_views)]
    return selected


def _build_perceptual_decoder(checkpoint: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, np.ndarray]:
    args = dict(checkpoint.get("args") or {})
    candidate_faces = np.asarray(checkpoint["candidate_faces"], dtype=np.int64)
    feature_mode = str(args.get("feature_mode", "basic"))
    model = PerceptualSurfaceResidualDecoder(
        int(candidate_faces.size),
        feature_dim=_decoder_feature_dim(feature_mode),
        embedding_dim=int(args.get("embedding_dim", 12)),
        hidden_dim=int(args.get("hidden_dim", 96)),
        layers=int(args.get("layers", 3)),
        max_delta=float(args.get("max_delta", 0.20)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, candidate_faces


def _predict_perceptual_delta_image(
    model: torch.nn.Module,
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    feature_mode: str,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    parent = np.asarray(z["rgb_render"], dtype=np.float32)
    delta = np.zeros_like(parent, dtype=np.float32)
    mask = _decoder_valid_mask(
        z,
        candidate_faces,
        residual_l1_key=str(residual_l1_key),
        min_l1=float(min_l1),
        min_alpha=float(min_alpha),
    )
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return delta
    faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
    face_idx, ok = _decoder_face_indices(faces, candidate_faces)
    ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
    with torch.no_grad():
        for start in range(0, int(ys.size), int(chunk_size)):
            end = min(int(ys.size), start + int(chunk_size))
            feat = torch.from_numpy(
                _decoder_load_feature_rows(z, ys[start:end], xs[start:end], feature_mode=str(feature_mode))
            ).to(device)
            face_t = torch.from_numpy(face_idx[start:end].astype(np.int64)).to(device)
            pred = model(face_t, feat).detach().cpu().numpy().astype(np.float32)
            delta[:, ys[start:end], xs[start:end]] = pred.T
    return delta


def _audit_policy_val(
    *,
    model: torch.nn.Module,
    face_lut: np.ndarray | None,
    model_kind: str,
    candidate_faces: np.ndarray | None,
    evidence_dir: Path,
    residual_rgb_key: str,
    residual_l1_key: str,
    alpha: float,
    alpha_conditioned_residual: bool,
    stride: int,
    max_views: int,
    min_alpha: float,
    min_l1: float,
    feature_mode: str,
    eval_tile: int,
    eval_overlap: int,
    eval_chunk_size: int,
    ssim_max_side: int,
    lpips_max_side: int,
    lpips_model: Any,
    region_grid: int,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for path in tqdm(_policy_val_paths(evidence_dir, stride, max_views), desc="policy-val checkpoint projection"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0).astype(np.float32)
            raw_delta = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
            teacher = np.clip(parent + raw_delta, 0.0, 1.0)
            active_mask = _valid_mask(z, float(min_alpha))
            if model_kind == "perceptual_surface_decoder":
                if candidate_faces is None:
                    raise RuntimeError("perceptual decoder audit requires candidate faces")
                active_mask = _decoder_valid_mask(
                    z,
                    candidate_faces,
                    residual_l1_key=str(residual_l1_key),
                    min_l1=float(min_l1),
                    min_alpha=float(min_alpha),
                )
                pred_delta = float(alpha) * _predict_perceptual_delta_image(
                    model,
                    z,
                    candidate_faces,
                    residual_l1_key=str(residual_l1_key),
                    min_l1=float(min_l1),
                    min_alpha=float(min_alpha),
                    feature_mode=str(feature_mode),
                    chunk_size=int(eval_chunk_size),
                    device=device,
                )
                support = {}
            else:
                features = torch.from_numpy(_load_input_chw(z))
                face_ids = _load_face_ids_tensor(z, face_lut, max_side=-1)
                if bool(alpha_conditioned_residual) and float(alpha) == 0.0:
                    pred_delta = np.zeros_like(parent, dtype=np.float32)
                    model_features = features
                else:
                    model_features = (
                        _append_alpha_channel_chw(features, float(alpha))
                        if bool(alpha_conditioned_residual)
                        else features
                    )
                    pred_delta_raw = _predict_delta_tiled(
                        model,
                        model_features,
                        face_ids=face_ids,
                        device=device,
                        tile=int(eval_tile),
                        overlap=int(eval_overlap),
                    )
                    pred_delta = pred_delta_raw.numpy()
                    if not bool(alpha_conditioned_residual):
                        pred_delta = float(alpha) * pred_delta
                support = _support_summary(model, model_features, face_ids, _valid_mask(z, float(min_alpha)))
            candidate = np.clip(parent + pred_delta, 0.0, 1.0)
            valid = _valid_mask(z, float(min_alpha))
            residual = _residual_stats(pred_delta, raw_delta, valid)
            active_residual = _residual_stats(pred_delta, raw_delta, active_mask)
            parent_lp = image_lpips_chw(parent, teacher, int(lpips_max_side), lpips_model) if lpips_model else None
            cand_lp = image_lpips_chw(candidate, teacher, int(lpips_max_side), lpips_model) if lpips_model else None
            row = {
                "view": path.stem,
                "parent_psnr": _psnr(parent, teacher),
                "candidate_psnr": _psnr(candidate, teacher),
                "parent_ssim": image_ssim_chw(parent, teacher, int(ssim_max_side)),
                "candidate_ssim": image_ssim_chw(candidate, teacher, int(ssim_max_side)),
                "parent_lpips": parent_lp,
                "candidate_lpips": cand_lp,
                **residual,
                **support,
            }
            rows.append(row)
            active_rows.append({"view": path.stem, **active_residual})
            for region in _region_rows(pred_delta, raw_delta, valid, int(region_grid)):
                region_rows.append({"view": path.stem, **region})
    return {
        "scope": "policy_val_teacher_residual",
        "image_summary": _summarize_image_rows(rows),
        "residual_summary": _summarize_residual_rows(rows),
        "active_residual_summary": _summarize_residual_rows(active_rows),
        "region_summary": _summarize_residual_rows(region_rows),
        "rows": rows,
        "active_rows": active_rows,
        "region_rows": region_rows,
    }


def _audit_target_apply(
    *,
    model: torch.nn.Module,
    face_lut: np.ndarray | None,
    target_eval_evidence_dir: Path,
    render_dir: Path,
    alpha: float,
    max_views: int,
    min_alpha: float,
    ssim_max_side: int,
    lpips_max_side: int,
    lpips_model: Any,
    region_grid: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    paths = evidence_views(target_eval_evidence_dir)
    if int(max_views) > 0:
        paths = paths[: int(max_views)]
    for path in tqdm(paths, desc="target final residual audit"):
        render_path = render_dir / f"{path.stem}.png"
        if not render_path.is_file():
            rows.append({"view": path.stem, "missing": True, "render_path": str(render_path)})
            continue
        with np.load(path, allow_pickle=False) as z:
            parent = np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0).astype(np.float32)
            gt = np.clip(_to_chw(z["rgb_gt"])[:3], 0.0, 1.0).astype(np.float32)
            final = _load_png_chw(render_path)
            final_delta = final - parent
            gt_delta = gt - parent
            valid = _valid_mask(z, float(min_alpha))
            residual = _residual_stats(final_delta, gt_delta, valid)
            parent_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if lpips_model else None
            final_lp = image_lpips_chw(final, gt, int(lpips_max_side), lpips_model) if lpips_model else None
            row = {
                "view": path.stem,
                "parent_psnr": _psnr(parent, gt),
                "candidate_psnr": _psnr(final, gt),
                "parent_ssim": image_ssim_chw(parent, gt, int(ssim_max_side)),
                "candidate_ssim": image_ssim_chw(final, gt, int(ssim_max_side)),
                "parent_lpips": parent_lp,
                "candidate_lpips": final_lp,
                "alpha": float(alpha),
                **residual,
            }
            rows.append(row)
            for region in _region_rows(final_delta, gt_delta, valid, int(region_grid)):
                region_rows.append({"view": path.stem, **region})
    return {
        "scope": "target_final_residual_vs_gt_after_apply",
        "render_dir": str(render_dir),
        "image_summary": _summarize_image_rows(rows),
        "residual_summary": _summarize_residual_rows(rows),
        "region_summary": _summarize_residual_rows(region_rows),
        "rows": rows,
        "region_rows": region_rows,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    policy = payload["policy_val_audit"]
    target = payload.get("target_apply_audit")
    lines = [
        "# Surface Checkpoint Residual Projection Audit",
        "",
        f"- run: `{payload['run_name']}`",
        f"- checkpoint: `{payload['inputs']['checkpoint']}`",
        f"- alpha: `{payload['inputs']['alpha']}`",
        f"- policy-val evidence: `{payload['inputs']['fit_evidence_dir']}`",
        "",
        "## Policy-Val Teacher Projection",
        "",
        "| scope | parent PSNR | candidate PSNR | PSNR gain | parent SSIM | candidate SSIM | SSIM gain | residual energy retention | cosine | changed fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ps = policy["image_summary"]
    rs = policy["residual_summary"]
    ars = policy.get("active_residual_summary", {})
    lines.append(
        "| policy-val vs teacher | {parent_psnr:.6f} | {cand_psnr:.6f} | {psnr_gain:+.6f} | {parent_ssim:.6f} | {cand_ssim:.6f} | {ssim_gain:+.6f} | {ret:.6f} | {cos:.6f} | {chg:.6f} |".format(
            parent_psnr=float(ps["parent_psnr"]),
            cand_psnr=float(ps["candidate_psnr"]),
            psnr_gain=float(ps["psnr_gain"]),
            parent_ssim=float(ps["parent_ssim"]),
            cand_ssim=float(ps["candidate_ssim"]),
            ssim_gain=float(ps["ssim_gain"]),
            ret=float(rs["energy_retention"]),
            cos=float(rs["cosine"]),
            chg=float(rs["changed_fraction"]),
        )
    )
    if ars:
        lines.append(
            "| selected-active residual only | {parent_psnr:.6f} | {cand_psnr:.6f} | {psnr_gain:+.6f} | {parent_ssim:.6f} | {cand_ssim:.6f} | {ssim_gain:+.6f} | {ret:.6f} | {cos:.6f} | {chg:.6f} |".format(
                parent_psnr=float(ps["parent_psnr"]),
                cand_psnr=float(ps["candidate_psnr"]),
                psnr_gain=float(ps["psnr_gain"]),
                parent_ssim=float(ps["parent_ssim"]),
                cand_ssim=float(ps["candidate_ssim"]),
                ssim_gain=float(ps["ssim_gain"]),
                ret=float(ars["energy_retention"]),
                cos=float(ars["cosine"]),
                chg=float(ars["changed_fraction"]),
            )
        )
    if target is not None:
        ts = target["image_summary"]
        trs = target["residual_summary"]
        lines.extend(
            [
                "",
                "## Target Final Residual",
                "",
                "| scope | parent PSNR | candidate PSNR | PSNR gain | parent SSIM | candidate SSIM | SSIM gain | residual energy retention | cosine | changed fraction |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| target vs GT | {parent_psnr:.6f} | {cand_psnr:.6f} | {psnr_gain:+.6f} | {parent_ssim:.6f} | {cand_ssim:.6f} | {ssim_gain:+.6f} | {ret:.6f} | {cos:.6f} | {chg:.6f} |".format(
                    parent_psnr=float(ts["parent_psnr"]),
                    cand_psnr=float(ts["candidate_psnr"]),
                    psnr_gain=float(ts["psnr_gain"]),
                    parent_ssim=float(ts["parent_ssim"]),
                    cand_ssim=float(ts["candidate_ssim"]),
                    ssim_gain=float(ts["ssim_gain"]),
                    ret=float(trs["energy_retention"]),
                    cos=float(trs["cosine"]),
                    chg=float(trs["changed_fraction"]),
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['output_json']}`",
            f"- Markdown: `{path}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit how a trained surface checkpoint projects teacher residuals.")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_type", choices=["surface_conditioned", "perceptual_surface_decoder"], default="surface_conditioned")
    parser.add_argument("--fit_evidence_dir", required=True)
    parser.add_argument("--target_eval_evidence_dir", default="")
    parser.add_argument("--target_render_dir", default="")
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--max_policy_views", type=int, default=0)
    parser.add_argument("--max_target_views", type=int, default=0)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--min_l1", type=float, default=0.0)
    parser.add_argument("--eval_tile", type=int, default=512)
    parser.add_argument("--eval_overlap", type=int, default=32)
    parser.add_argument("--eval_chunk_size", type=int, default=65536)
    parser.add_argument("--ssim_max_side", type=int, default=512)
    parser.add_argument("--lpips_max_side", type=int, default=256)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--region_grid", type=int, default=4)
    parser.add_argument("--keep_rows", action="store_true")
    parser.add_argument("--worst_region_count", type=int, default=12)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    checkpoint_args = dict(checkpoint.get("args") or {})
    alpha_conditioned_residual = bool(checkpoint_args.get("alpha_conditioned_residual", False))
    perceptual_feature_mode = str(checkpoint_args.get("feature_mode", "basic"))
    candidate_faces = None
    if str(args.checkpoint_type) == "perceptual_surface_decoder":
        model, candidate_faces = _build_perceptual_decoder(checkpoint, device)
        face_lut = None
    else:
        model, face_lut = _build_model(checkpoint, device)
    lpips_model = build_lpips_model() if bool(args.compute_lpips) else None

    policy = _audit_policy_val(
        model=model,
        face_lut=face_lut,
        model_kind=str(args.checkpoint_type),
        candidate_faces=candidate_faces,
        evidence_dir=Path(args.fit_evidence_dir),
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        alpha=float(args.alpha),
        alpha_conditioned_residual=bool(alpha_conditioned_residual),
        stride=int(args.policy_val_stride),
        max_views=int(args.max_policy_views),
        min_alpha=float(args.min_alpha),
        min_l1=float(args.min_l1),
        feature_mode=perceptual_feature_mode,
        eval_tile=int(args.eval_tile),
        eval_overlap=int(args.eval_overlap),
        eval_chunk_size=int(args.eval_chunk_size),
        ssim_max_side=int(args.ssim_max_side),
        lpips_max_side=int(args.lpips_max_side),
        lpips_model=lpips_model,
        region_grid=int(args.region_grid),
        device=device,
    )
    target = None
    if str(args.target_eval_evidence_dir) and str(args.target_render_dir):
        target = _audit_target_apply(
            model=model,
            face_lut=face_lut,
            target_eval_evidence_dir=Path(args.target_eval_evidence_dir),
            render_dir=Path(args.target_render_dir),
            alpha=float(args.alpha),
            max_views=int(args.max_target_views),
            min_alpha=float(args.min_alpha),
            ssim_max_side=int(args.ssim_max_side),
            lpips_max_side=int(args.lpips_max_side),
            lpips_model=lpips_model,
            region_grid=int(args.region_grid),
        )

    policy_resid = policy["residual_summary"]
    target_resid = {} if target is None else target["residual_summary"]
    if float(policy_resid.get("energy_retention", 0.0)) < 0.2:
        interpretation = (
            "The checkpoint retains very little teacher residual energy on policy-val views. "
            "This points to carrier/objective underfitting before target transfer."
        )
    elif float(policy_resid.get("cosine", 0.0)) < 0.2:
        interpretation = (
            "The checkpoint writes nontrivial energy, but it is poorly aligned with the teacher residual. "
            "This points to representation or loss mismatch rather than a capacity-only issue."
        )
    elif target is not None and float(target_resid.get("cosine", 0.0)) < 0.2:
        interpretation = (
            "The checkpoint can mimic teacher residuals on policy-val views, but target final residuals are weakly aligned "
            "with target GT residuals. This points to cross-view transfer/certification failure."
        )
    else:
        interpretation = (
            "Residual projection is not the dominant measured failure under this audit. The next check should focus on "
            "target masking, clipping, alpha selection, and official metric tails."
        )

    payload: dict[str, Any] = {
        "schema": "spcarnet_surface_checkpoint_residual_projection_audit_v1",
        "run_name": str(args.run_name),
        "inputs": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_type": str(args.checkpoint_type),
            "fit_evidence_dir": str(args.fit_evidence_dir),
            "target_eval_evidence_dir": str(args.target_eval_evidence_dir),
            "target_render_dir": str(args.target_render_dir),
            "residual_rgb_key": str(args.residual_rgb_key),
            "residual_l1_key": str(args.residual_l1_key),
            "perceptual_feature_mode": str(perceptual_feature_mode),
            "alpha": float(args.alpha),
            "alpha_conditioned_residual": bool(alpha_conditioned_residual),
            "alpha_contract": (
                "model_outputs_final_delta_for_selected_alpha"
                if bool(alpha_conditioned_residual)
                else "posthoc_policy_val_alpha_multiplier"
            ),
            "policy_val_stride": int(args.policy_val_stride),
            "min_alpha": float(args.min_alpha),
            "min_l1": float(args.min_l1),
            "region_grid": int(args.region_grid),
        },
        "checkpoint_args": checkpoint.get("args", {}),
        "policy_val_audit": _compact_audit(policy, bool(args.keep_rows), int(args.worst_region_count)),
        "target_apply_audit": None if target is None else _compact_audit(target, bool(args.keep_rows), int(args.worst_region_count)),
        "interpretation": interpretation,
        "output_json": str(args.output_json),
    }
    _write_json(Path(args.output_json), payload)
    _write_markdown(Path(args.output_md), payload)
    _write_json(Path(args.output_json), payload)
    print(json.dumps({
        "run_name": str(args.run_name),
        "policy_energy_retention": float(policy_resid.get("energy_retention", 0.0)),
        "policy_cosine": float(policy_resid.get("cosine", 0.0)),
        "target_energy_retention": None if target is None else float(target_resid.get("energy_retention", 0.0)),
        "target_cosine": None if target is None else float(target_resid.get("cosine", 0.0)),
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
