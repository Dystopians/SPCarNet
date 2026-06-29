#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import (  # noqa: E402
    build_lpips_model,
    evidence_views,
    image_lpips_chw,
    image_ssim_chw,
    save_image_chw,
)
from scripts.car_model.train_perceptual_surface_residual_decoder import (  # noqa: E402
    DEFAULT_EVIDENCE,
    _face_indices,
    _policy_split,
    _rank_candidate_faces,
    _valid_mask,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean(np.square(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))
    return float("inf") if mse <= 1.0e-12 else float(-10.0 * math.log10(mse))


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


def _bin_ids(z: np.lib.npyio.NpzFile, ys: np.ndarray, xs: np.ndarray, grid: int) -> np.ndarray:
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    u = np.clip(bary[1, ys, xs], 0.0, 1.0 - 1.0e-6)
    v = np.clip(bary[2, ys, xs], 0.0, 1.0 - 1.0e-6)
    iu = np.floor(u * int(grid)).astype(np.int64)
    iv = np.floor(v * int(grid)).astype(np.int64)
    return iu * int(grid) + iv


def _fit_texture(
    fit_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    grid: int,
    ridge_count: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    bins = int(grid) * int(grid)
    sums = np.zeros((int(candidate_faces.size), bins, 3), dtype=np.float64)
    counts = np.zeros((int(candidate_faces.size), bins), dtype=np.float64)
    active_pixels = 0
    used_pixels = 0
    for path in tqdm(fit_paths, desc="fit UV residual texture"):
        with np.load(path, allow_pickle=False) as z:
            mask = _valid_mask(
                z,
                candidate_faces,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
            )
            ys, xs = np.nonzero(mask)
            active_pixels += int(ys.size)
            if ys.size == 0:
                continue
            faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
            face_idx, ok = _face_indices(faces, candidate_faces)
            if not np.any(ok):
                continue
            ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
            bin_id = _bin_ids(z, ys, xs, int(grid))
            residual = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
            np.add.at(counts, (face_idx, bin_id), 1.0)
            for c in range(3):
                np.add.at(sums[:, :, c], (face_idx, bin_id), residual[c, ys, xs].astype(np.float64))
            used_pixels += int(ys.size)
    denom = counts[:, :, None] + float(ridge_count)
    texture = np.divide(sums, np.maximum(denom, 1.0e-12), out=np.zeros_like(sums), where=denom > 0.0).astype(np.float32)
    return texture, counts.astype(np.float32), {
        "fit_active_pixels": int(active_pixels),
        "fit_used_pixels": int(used_pixels),
        "nonempty_bins": int(np.count_nonzero(counts > 0.0)),
        "nonempty_bin_fraction": float(np.mean(counts > 0.0)),
        "mean_nonempty_count": float(np.mean(counts[counts > 0.0])) if np.any(counts > 0.0) else 0.0,
        "ridge_count": float(ridge_count),
    }


def _predict_delta(
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray,
    texture: np.ndarray,
    counts: np.ndarray,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    min_bin_count: float,
    grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(z["rgb_render"], dtype=np.float32)
    delta = np.zeros_like(parent, dtype=np.float32)
    active = _valid_mask(
        z,
        candidate_faces,
        residual_l1_key=str(residual_l1_key),
        min_l1=float(min_l1),
        min_alpha=float(min_alpha),
    )
    ys, xs = np.nonzero(active)
    if ys.size == 0:
        return delta, active
    faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
    face_idx, ok = _face_indices(faces, candidate_faces)
    ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
    if ys.size == 0:
        return delta, active
    bin_id = _bin_ids(z, ys, xs, int(grid))
    good = counts[face_idx, bin_id] >= float(min_bin_count)
    ys, xs, face_idx, bin_id = ys[good], xs[good], face_idx[good], bin_id[good]
    if ys.size:
        delta[:, ys, xs] = texture[face_idx, bin_id].T
    return delta, active


def _evaluate(
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    texture: np.ndarray,
    counts: np.ndarray,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    min_bin_count: float,
    grid: int,
    alpha_grid: list[float],
    compute_lpips: bool,
    ssim_max_side: int,
    lpips_max_side: int,
    output_dir: Path,
) -> dict[str, Any]:
    lpips_model = build_lpips_model() if compute_lpips else None
    rows_by_alpha: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    projection_rows: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    active_projection_rows: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(val_paths, desc="eval UV residual texture"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            raw_delta = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
            teacher = np.clip(parent + raw_delta, 0.0, 1.0)
            delta, active_mask = _predict_delta(
                z,
                candidate_faces,
                texture,
                counts,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
                min_bin_count=float(min_bin_count),
                grid=int(grid),
            )
            full_mask = np.asarray(z["face_id"], dtype=np.int64) >= 0
            if "barycentric_valid" in z:
                full_mask &= np.asarray(z["barycentric_valid"]).astype(bool)
            if "alpha" in z:
                full_mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
            p_psnr = _psnr(parent, gt)
            p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
            p_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
            t_parent_psnr = _psnr(parent, teacher)
            t_parent_ssim = image_ssim_chw(parent, teacher, int(ssim_max_side))
            t_parent_lp = image_lpips_chw(parent, teacher, int(lpips_max_side), lpips_model) if compute_lpips else None
            for alpha in alpha_grid:
                pred = float(alpha) * delta
                adapted = np.clip(parent + pred, 0.0, 1.0)
                c_psnr = _psnr(adapted, gt)
                c_ssim = image_ssim_chw(adapted, gt, int(ssim_max_side))
                c_lp = image_lpips_chw(adapted, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
                rows_by_alpha[float(alpha)].append(
                    {
                        "view": path.stem,
                        "parent_psnr": float(p_psnr),
                        "candidate_psnr": float(c_psnr),
                        "psnr_gain": float(c_psnr - p_psnr),
                        "parent_ssim": float(p_ssim),
                        "candidate_ssim": float(c_ssim),
                        "ssim_gain": float(c_ssim - p_ssim),
                        "parent_lpips": float(p_lp) if compute_lpips else None,
                        "candidate_lpips": float(c_lp) if compute_lpips else None,
                        "lpips_gain": float(p_lp - c_lp) if compute_lpips else None,
                    }
                )
                t_cand_psnr = _psnr(adapted, teacher)
                t_cand_ssim = image_ssim_chw(adapted, teacher, int(ssim_max_side))
                t_cand_lp = image_lpips_chw(adapted, teacher, int(lpips_max_side), lpips_model) if compute_lpips else None
                projection_rows[float(alpha)].append(
                    {
                        "view": path.stem,
                        "parent_psnr": float(t_parent_psnr),
                        "candidate_psnr": float(t_cand_psnr),
                        "psnr_gain": float(t_cand_psnr - t_parent_psnr),
                        "parent_ssim": float(t_parent_ssim),
                        "candidate_ssim": float(t_cand_ssim),
                        "ssim_gain": float(t_cand_ssim - t_parent_ssim),
                        "parent_lpips": float(t_parent_lp) if compute_lpips else None,
                        "candidate_lpips": float(t_cand_lp) if compute_lpips else None,
                        "lpips_gain": float(t_parent_lp - t_cand_lp) if compute_lpips else None,
                        **_residual_stats(pred, raw_delta, full_mask),
                    }
                )
                active_projection_rows[float(alpha)].append({"view": path.stem, **_residual_stats(pred, raw_delta, active_mask)})
    summaries: list[dict[str, Any]] = []
    projection_summaries: dict[str, Any] = {}
    for alpha, rows in rows_by_alpha.items():
        psnr_gain = [float(r["psnr_gain"]) for r in rows]
        ssim_gain = [float(r["ssim_gain"]) for r in rows]
        lpips_gain = [float(r["lpips_gain"]) for r in rows if r.get("lpips_gain") is not None]
        row = {
            "alpha": float(alpha),
            "parent_psnr": _mean([float(r["parent_psnr"]) for r in rows]),
            "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in rows]),
            "psnr_gain": _mean(psnr_gain),
            "parent_ssim": _mean([float(r["parent_ssim"]) for r in rows]),
            "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in rows]),
            "ssim_gain": _mean(ssim_gain),
            "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)),
            "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)),
        }
        if compute_lpips:
            row.update(
                {
                    "parent_lpips": _mean([float(r["parent_lpips"]) for r in rows]),
                    "candidate_lpips": _mean([float(r["candidate_lpips"]) for r in rows]),
                    "lpips_gain": _mean(lpips_gain),
                    "lpips_positive_view_fraction": float(np.mean(np.asarray(lpips_gain) > 0.0)),
                }
            )
        summaries.append(row)
        proj_rows = projection_rows[alpha]
        proj_img = {
            "parent_psnr": _mean([float(r["parent_psnr"]) for r in proj_rows]),
            "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in proj_rows]),
            "psnr_gain": _mean([float(r["psnr_gain"]) for r in proj_rows]),
            "parent_ssim": _mean([float(r["parent_ssim"]) for r in proj_rows]),
            "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in proj_rows]),
            "ssim_gain": _mean([float(r["ssim_gain"]) for r in proj_rows]),
            "parent_lpips": _mean([float(r["parent_lpips"]) for r in proj_rows if r.get("parent_lpips") is not None]),
            "candidate_lpips": _mean([float(r["candidate_lpips"]) for r in proj_rows if r.get("candidate_lpips") is not None]),
            "lpips_gain": _mean([float(r["lpips_gain"]) for r in proj_rows if r.get("lpips_gain") is not None]),
        }
        projection_summaries[str(alpha)] = {
            "image_summary": proj_img,
            "full_residual_summary": _summarize_residual_rows(proj_rows),
            "active_residual_summary": _summarize_residual_rows(active_projection_rows[alpha]),
        }
    best = max(
        summaries,
        key=lambda r: float(r.get("psnr_gain", 0.0))
        + 20.0 * float(r.get("ssim_gain", 0.0))
        + 20.0 * float(r.get("lpips_gain", 0.0)),
    )
    best_all_axis = None
    for row in summaries:
        if (
            float(row.get("psnr_gain", 0.0)) > 0.0
            and float(row.get("ssim_gain", 0.0)) > 0.0
            and (not compute_lpips or float(row.get("lpips_gain", 0.0)) > 0.0)
        ):
            cand = dict(row)
            cand["balanced_score"] = (
                float(row.get("psnr_gain", 0.0))
                + 20.0 * float(row.get("ssim_gain", 0.0))
                + 20.0 * float(row.get("lpips_gain", 0.0))
            )
            if best_all_axis is None or float(cand["balanced_score"]) > float(best_all_axis["balanced_score"]):
                best_all_axis = cand
    best_alpha = float((best_all_axis or best)["alpha"])
    render_dir = output_dir / "policy_val_best"
    render_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(val_paths, desc="write UV policy-val renders"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            delta, _ = _predict_delta(
                z,
                candidate_faces,
                texture,
                counts,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
                min_bin_count=float(min_bin_count),
                grid=int(grid),
            )
            save_image_chw(render_dir / f"{path.stem}.png", np.clip(parent + best_alpha * delta, 0.0, 1.0))
            save_image_chw(render_dir / f"{path.stem}_gt.png", gt)
    return {
        "best": best,
        "best_all_axis": best_all_axis,
        "rows": summaries,
        "projection_by_alpha": projection_summaries,
        "best_alpha_for_render": best_alpha,
        "render_dir": str(render_dir),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload["policy_val"]["best"]
    best_all = payload["policy_val"].get("best_all_axis")
    alpha = str(float((best_all or best)["alpha"]))
    proj = payload["policy_val"]["projection_by_alpha"][alpha]
    lines = [
        "# v209 Surface UV Residual Texture Audit",
        "",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- selected faces: `{payload['candidate_face_summary']['selected_faces']}`",
        f"- selected score coverage: `{payload['candidate_face_summary'].get('selected_score_coverage', 0.0):.6f}`",
        f"- grid: `{payload['fit']['grid']}`",
        f"- best all-axis: `{best_all}`",
        "",
        "## Projection At Selected Alpha",
        "",
        "| scope | PSNR gain vs teacher | SSIM gain vs teacher | LPIPS gain vs teacher | cosine | energy retention | changed fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            "| full valid | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {cos:.6f} | {ret:.6f} | {chg:.6f} |"
        ).format(
            psnr=float(proj["image_summary"]["psnr_gain"]),
            ssim=float(proj["image_summary"]["ssim_gain"]),
            lpips=float(proj["image_summary"].get("lpips_gain", 0.0)),
            cos=float(proj["full_residual_summary"]["cosine"]),
            ret=float(proj["full_residual_summary"]["energy_retention"]),
            chg=float(proj["full_residual_summary"]["changed_fraction"]),
        ),
        (
            "| selected active | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {cos:.6f} | {ret:.6f} | {chg:.6f} |"
        ).format(
            psnr=float(proj["image_summary"]["psnr_gain"]),
            ssim=float(proj["image_summary"]["ssim_gain"]),
            lpips=float(proj["image_summary"].get("lpips_gain", 0.0)),
            cos=float(proj["active_residual_summary"]["cosine"]),
            ret=float(proj["active_residual_summary"]["energy_retention"]),
            chg=float(proj["active_residual_summary"]["changed_fraction"]),
        ),
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- renders: `{payload['policy_val']['render_dir']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit an explicit per-face UV teacher-residual micro-texture.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--min_l1", type=float, default=0.0005)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--max_candidate_faces", type=int, default=131072)
    parser.add_argument("--candidate_target_energy_coverage", type=float, default=0.90)
    parser.add_argument("--max_candidate_face_samples_per_view", type=int, default=65536)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--ridge_count", type=float, default=4.0)
    parser.add_argument("--min_bin_count", type=float, default=2.0)
    parser.add_argument("--alpha_grid", default="0,0.03125,0.0625,0.125,0.25,0.5,0.75,1")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--policy_val_ssim_max_side", type=int, default=512)
    parser.add_argument("--policy_val_lpips_max_side", type=int, default=256)
    parser.add_argument("--output_dir", default="/tmp/peilincai_spcarnet_v209_uv_residual_texture")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-v209-uv-residual-texture")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--seed", type=int, default=209)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if bool(args.enable_wandb):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_run_name or output_dir.name),
                config=vars(args),
                dir=str(output_dir),
            )
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] disabled after init failure: {type(exc).__name__}: {exc}", flush=True)
            wandb_run = None

    paths = evidence_views(Path(args.fit_evidence_dir))
    if not paths:
        raise FileNotFoundError(args.fit_evidence_dir)
    fit_paths, val_paths = _policy_split(paths, int(args.policy_val_stride))
    candidate_faces, face_summary = _rank_candidate_faces(
        fit_paths,
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        max_faces=int(args.max_candidate_faces),
        max_samples_per_view=int(args.max_candidate_face_samples_per_view),
        target_energy_coverage=float(args.candidate_target_energy_coverage),
        seed=int(args.seed),
    )
    texture, counts, fit_summary = _fit_texture(
        fit_paths,
        candidate_faces,
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        grid=int(args.grid),
        ridge_count=float(args.ridge_count),
    )
    alpha_grid = sorted({float(x) for x in str(args.alpha_grid).split(",") if x.strip()})
    policy_val = _evaluate(
        val_paths,
        candidate_faces,
        texture,
        counts,
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        min_bin_count=float(args.min_bin_count),
        grid=int(args.grid),
        alpha_grid=alpha_grid,
        compute_lpips=bool(args.compute_lpips),
        ssim_max_side=int(args.policy_val_ssim_max_side),
        lpips_max_side=int(args.policy_val_lpips_max_side),
        output_dir=output_dir,
    )
    all_axis = policy_val.get("best_all_axis") is not None
    checkpoint = {
        "schema": "spcarnet_uv_residual_texture_checkpoint_v1",
        "candidate_faces": candidate_faces,
        "texture": texture.astype(np.float16),
        "counts": counts.astype(np.float16),
        "args_json": json.dumps(vars(args), sort_keys=True),
    }
    checkpoint_path = output_dir / "v209_uv_residual_texture.npz"
    np.savez_compressed(checkpoint_path, **checkpoint)
    payload: dict[str, Any] = {
        "schema": "spcarnet_uv_residual_texture_audit_v1",
        "created_at": "2026-06-29",
        "fit_evidence_dir": str(args.fit_evidence_dir),
        "fit_views": int(len(fit_paths)),
        "policy_val_views": int(len(val_paths)),
        "candidate_face_summary": face_summary,
        "fit": {
            **fit_summary,
            "grid": int(args.grid),
            "bins_per_face": int(args.grid) * int(args.grid),
            "min_bin_count": float(args.min_bin_count),
        },
        "policy_val_all_axis_pass": bool(all_axis),
        "policy_val": policy_val,
        "uses_train_fit_teacher": True,
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "checkpoint": str(checkpoint_path),
        "output_json": str(output_dir / "v209_uv_residual_texture_audit.json"),
    }
    _write_json(output_dir / "v209_uv_residual_texture_audit.json", payload)
    _write_md(output_dir / "v209_uv_residual_texture_audit.md", payload)
    if wandb_run is not None:
        best = policy_val["best"]
        best_all = policy_val.get("best_all_axis") or {}
        selected_alpha = str(float((best_all or best)["alpha"]))
        proj = policy_val["projection_by_alpha"][selected_alpha]
        wandb_run.log(
            {
                "policy_val/all_axis_pass": int(all_axis),
                "policy_val/best_alpha": float(best["alpha"]),
                "policy_val/best_psnr_gain": float(best.get("psnr_gain", 0.0)),
                "policy_val/best_ssim_gain": float(best.get("ssim_gain", 0.0)),
                "policy_val/best_lpips_gain": float(best.get("lpips_gain", 0.0)),
                "projection/full_cosine": float(proj["full_residual_summary"]["cosine"]),
                "projection/full_energy_retention": float(proj["full_residual_summary"]["energy_retention"]),
                "projection/active_cosine": float(proj["active_residual_summary"]["cosine"]),
                "projection/active_energy_retention": float(proj["active_residual_summary"]["energy_retention"]),
            }
        )
        wandb_run.finish()
    print(
        json.dumps(
            {
                "output_json": str(output_dir / "v209_uv_residual_texture_audit.json"),
                "output_md": str(output_dir / "v209_uv_residual_texture_audit.md"),
                "policy_val_all_axis_pass": bool(all_axis),
                "best": policy_val["best"],
                "best_all_axis": policy_val.get("best_all_axis"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
