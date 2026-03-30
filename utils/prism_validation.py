import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.geometry_metrics_utils import depth_metrics, normal_metrics_from_abs_cos


@dataclass
class PrismValidationConfig:
    interval: int = 1000
    max_views: int = 32
    absrel_rel_degrade_thresh: float = 0.01
    mean_angle_degrade_thresh_deg: float = 0.4
    psnr_drop_thresh_db: float = 0.10
    mae_increase_thresh: float = 0.003


def _normalize_name(name: str) -> str:
    base = os.path.basename(name)
    return os.path.splitext(base)[0].lower()


def _load_split_dropped_keys(split_file: str) -> List[str]:
    if (not split_file) or (not os.path.exists(split_file)):
        return []
    try:
        with open(split_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return [_normalize_name(x) for x in payload.get("dropped", [])]
    except Exception:
        return []


def _build_all_colmap_cameras(dataset) -> List:
    if not os.path.exists(os.path.join(dataset.source_path, "sparse")):
        return []
    try:
        info = sceneLoadTypeCallbacks["Colmap"](
            dataset.source_path,
            dataset.images,
            False,
            split_strategy="llff",
            split_file="",
        )
        return cameraList_from_camInfos(info.train_cameras, 1.0, dataset)
    except Exception:
        return []


def build_prism_validation_views(scene, dataset, cfg: PrismValidationConfig) -> List:
    """
    Dev validation set for PRISM:
    - Prefer dropped buffer views from split file (when available).
    - Keep test set untouched (never used as primary PRISM dev-val under file split with dropped).
    """
    max_views = max(1, int(cfg.max_views))
    selected: List = []

    use_file_split = str(getattr(dataset, "split_strategy", "")).strip().lower() == "file"
    dropped = _load_split_dropped_keys(str(getattr(dataset, "split_file", ""))) if use_file_split else []
    if len(dropped) > 0:
        all_cams = _build_all_colmap_cameras(dataset)
        by_name = {_normalize_name(getattr(c, "image_name", "")): c for c in all_cams}
        for k in dropped:
            cam = by_name.get(k, None)
            if cam is None:
                continue
            selected.append(cam)
            if len(selected) >= max_views:
                break
        if len(selected) > 0:
            return selected

    # Fallback only when dropped buffer is unavailable.
    for cam in scene.getTestCameras():
        selected.append(cam)
        if len(selected) >= max_views:
            break
    return selected


def _safe_mean(values: List[float]) -> float:
    vals = [v for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def evaluate_prism_validation_metrics(
    views: Sequence,
    triangles,
    render_func,
    pipe,
    background: torch.Tensor,
) -> Dict[str, float]:
    psnr_vals = []
    mae_vals = []
    absrel_vals = []
    delta_vals = []
    mean_angle_vals = []
    abs_cos_vals = []

    roi_psnr_vals = []
    roi_mae_vals = []
    roi_absrel_vals = []
    roi_mean_angle_vals = []
    roi_abs_cos_vals = []

    with torch.no_grad():
        for v in views:
            pkg = render_func(v, triangles, pipe, background)
            img = torch.clamp(pkg["render"], 0.0, 1.0)
            gt = torch.clamp(v.original_image.to(img.device), 0.0, 1.0)
            diff = img - gt
            mae = float(torch.mean(torch.abs(diff)).item())
            mse = float(torch.mean(diff * diff).item())
            psnr = float(-10.0 * np.log10(max(mse, 1e-8)))
            mae_vals.append(mae)
            psnr_vals.append(psnr)

            # Depth metrics (if available)
            absrel = float("nan")
            delta_125 = float("nan")
            if getattr(v, "invdepthmap", None) is not None:
                pred_depth = pkg["surf_depth"].detach()[0]
                gt_inv = v.invdepthmap.to(pred_depth.device)[0]
                gt_depth = 1.0 / torch.clamp(gt_inv, min=1e-6)
                valid = torch.isfinite(gt_depth) & torch.isfinite(pred_depth) & (gt_depth > 1e-6) & (pred_depth > 1e-6)
                if torch.any(valid):
                    dm = depth_metrics(
                        pred=pred_depth[valid].detach().cpu().numpy().astype(np.float64),
                        gt=gt_depth[valid].detach().cpu().numpy().astype(np.float64),
                    )
                    absrel = float(dm["abs_rel"])
                    delta_125 = float(dm["delta_1.25"])
            absrel_vals.append(absrel)
            delta_vals.append(delta_125)

            # Normal metrics (if available)
            mean_ang = float("nan")
            abs_cos = float("nan")
            if getattr(v, "normal_map", None) is not None:
                rn = F.normalize(pkg["rend_normal"].detach(), dim=0, eps=1e-6)
                gn = F.normalize(v.normal_map.to(rn.device), dim=0, eps=1e-6)
                c = torch.clamp(torch.abs(torch.sum(rn * gn, dim=0)), 0.0, 1.0)
                nm = normal_metrics_from_abs_cos(c.detach().cpu().numpy().astype(np.float64))
                mean_ang = float(nm["mean_ang_deg"])
                abs_cos = float(nm["mean_abs_cos"])
            mean_angle_vals.append(mean_ang)
            abs_cos_vals.append(abs_cos)

            # Optional ROI breakdown from existing ground mask (analysis only).
            gm = getattr(v, "ground_mask", None)
            if gm is not None:
                mask = gm.to(img.device)
                if mask.ndim == 3:
                    mask = mask[0]
                if mask.ndim != 2:
                    continue
                if mask.shape[0] != img.shape[1] or mask.shape[1] != img.shape[2]:
                    mask = F.interpolate(mask[None, None].float(), size=(img.shape[1], img.shape[2]), mode="nearest").squeeze(0).squeeze(0)
                mask = mask > 0.5
                if torch.any(mask):
                    mask3 = mask.unsqueeze(0).float()
                    roi_mae = float((torch.abs(diff) * mask3).sum().item() / (mask3.sum().item() * 3.0 + 1e-8))
                    roi_mse = float(((diff * diff) * mask3).sum().item() / (mask3.sum().item() * 3.0 + 1e-8))
                    roi_psnr = float(-10.0 * np.log10(max(roi_mse, 1e-8)))
                    roi_mae_vals.append(roi_mae)
                    roi_psnr_vals.append(roi_psnr)

                    if getattr(v, "invdepthmap", None) is not None:
                        pred_depth = pkg["surf_depth"].detach()[0]
                        gt_inv = v.invdepthmap.to(pred_depth.device)[0]
                        gt_depth = 1.0 / torch.clamp(gt_inv, min=1e-6)
                        valid = mask & torch.isfinite(gt_depth) & torch.isfinite(pred_depth) & (gt_depth > 1e-6) & (pred_depth > 1e-6)
                        if torch.any(valid):
                            dm_roi = depth_metrics(
                                pred=pred_depth[valid].detach().cpu().numpy().astype(np.float64),
                                gt=gt_depth[valid].detach().cpu().numpy().astype(np.float64),
                            )
                            roi_absrel_vals.append(float(dm_roi["abs_rel"]))

                    if getattr(v, "normal_map", None) is not None:
                        rn = F.normalize(pkg["rend_normal"].detach(), dim=0, eps=1e-6)
                        gn = F.normalize(v.normal_map.to(rn.device), dim=0, eps=1e-6)
                        c = torch.clamp(torch.abs(torch.sum(rn * gn, dim=0)), 0.0, 1.0)
                        c_roi = c[mask]
                        if c_roi.numel() > 0:
                            nm_roi = normal_metrics_from_abs_cos(c_roi.detach().cpu().numpy().astype(np.float64))
                            roi_mean_angle_vals.append(float(nm_roi["mean_ang_deg"]))
                            roi_abs_cos_vals.append(float(nm_roi["mean_abs_cos"]))

    return {
        "num_views": float(len(views)),
        "psnr": _safe_mean(psnr_vals),
        "mae": _safe_mean(mae_vals),
        "absrel": _safe_mean(absrel_vals),
        "delta_1.25": _safe_mean(delta_vals),
        "mean_angle": _safe_mean(mean_angle_vals),
        "abs_cos": _safe_mean(abs_cos_vals),
        # Optional analysis-only ROI breakdown
        "roi_psnr": _safe_mean(roi_psnr_vals),
        "roi_mae": _safe_mean(roi_mae_vals),
        "roi_absrel": _safe_mean(roi_absrel_vals),
        "roi_mean_angle": _safe_mean(roi_mean_angle_vals),
        "roi_abs_cos": _safe_mean(roi_abs_cos_vals),
    }


def compare_validation_against_stage_best(
    current_metrics: Dict[str, float],
    stage_best_metrics: Dict[str, float],
    cfg: PrismValidationConfig,
) -> Tuple[bool, Dict[str, float], List[str]]:
    """
    Return:
    - pass_gate
    - deltas
    - triggered_rules
    """
    deltas = {
        "absrel_rel_degrade": float("nan"),
        "mean_angle_degrade": float("nan"),
        "psnr_drop": float("nan"),
        "mae_increase": float("nan"),
    }
    rules = []

    cur_absrel = current_metrics.get("absrel", float("nan"))
    bst_absrel = stage_best_metrics.get("absrel", float("nan"))
    if np.isfinite(cur_absrel) and np.isfinite(bst_absrel) and bst_absrel > 1e-8:
        rel_deg = (cur_absrel - bst_absrel) / bst_absrel
        deltas["absrel_rel_degrade"] = float(rel_deg)
        if rel_deg > float(cfg.absrel_rel_degrade_thresh):
            rules.append("absrel_rel_degrade")

    cur_ang = current_metrics.get("mean_angle", float("nan"))
    bst_ang = stage_best_metrics.get("mean_angle", float("nan"))
    if np.isfinite(cur_ang) and np.isfinite(bst_ang):
        d = cur_ang - bst_ang
        deltas["mean_angle_degrade"] = float(d)
        if d > float(cfg.mean_angle_degrade_thresh_deg):
            rules.append("mean_angle_degrade")

    cur_psnr = current_metrics.get("psnr", float("nan"))
    bst_psnr = stage_best_metrics.get("psnr", float("nan"))
    if np.isfinite(cur_psnr) and np.isfinite(bst_psnr):
        drop = bst_psnr - cur_psnr
        deltas["psnr_drop"] = float(drop)
        if drop > float(cfg.psnr_drop_thresh_db):
            rules.append("psnr_drop")

    cur_mae = current_metrics.get("mae", float("nan"))
    bst_mae = stage_best_metrics.get("mae", float("nan"))
    if np.isfinite(cur_mae) and np.isfinite(bst_mae):
        inc = cur_mae - bst_mae
        deltas["mae_increase"] = float(inc)
        if inc > float(cfg.mae_increase_thresh):
            rules.append("mae_increase")

    return (len(rules) == 0), deltas, rules


def save_validation_summary(
    out_dir: str,
    iteration: int,
    phase_name: str,
    current_metrics: Dict[str, float],
    stage_best_metrics: Dict[str, float],
    deltas: Dict[str, float],
    pass_gate: bool,
    triggered_rules: Sequence[str],
):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "iteration": int(iteration),
        "phase": str(phase_name),
        "pass_gate": bool(pass_gate),
        "triggered_rules": list(triggered_rules),
        "current": current_metrics,
        "stage_best": stage_best_metrics,
        "deltas": deltas,
    }
    json_path = os.path.join(out_dir, f"validation_iter_{int(iteration):06d}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.join(out_dir, f"validation_iter_{int(iteration):06d}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PRISM Validation Summary\n\n")
        f.write(f"- iteration: {int(iteration)}\n")
        f.write(f"- phase: {phase_name}\n")
        f.write(f"- pass_gate: {bool(pass_gate)}\n")
        f.write(f"- triggered_rules: {', '.join(triggered_rules) if len(triggered_rules) > 0 else 'none'}\n\n")
        f.write("## Current Metrics\n")
        for k, v in current_metrics.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Stage-Best Metrics\n")
        for k, v in stage_best_metrics.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Deltas\n")
        for k, v in deltas.items():
            f.write(f"- {k}: {v}\n")
