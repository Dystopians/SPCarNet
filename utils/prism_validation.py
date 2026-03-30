import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.prism_geometry_proxy import (
    GeometryProxyConfig,
    GeometryProxyContext,
    estimate_view_sparse_observability,
    evaluate_view_sparse_geometry_proxy,
    normalize_image_key,
)


@dataclass
class PrismValidationConfig:
    interval: int = 1000
    max_views: int = 32
    absrel_rel_degrade_thresh: float = 0.01
    mean_angle_degrade_thresh_deg: float = 0.4
    psnr_drop_thresh_db: float = 0.10
    mae_increase_thresh: float = 0.003
    # Hybrid view construction
    num_buffer_views: int = 16
    num_train_views: int = 16
    train_pool_size: int = 128
    prefer_observable_train_views: bool = True
    min_depth_matches_per_view: int = 24
    min_normal_matches_per_view: int = 8
    # Gate observability thresholds
    min_valid_depth_matches: int = 128
    min_valid_normal_matches: int = 64


def _load_split_dropped_keys(split_file: str) -> List[str]:
    if (not split_file) or (not os.path.exists(split_file)):
        return []
    try:
        with open(split_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return [normalize_image_key(x) for x in payload.get("dropped", [])]
    except Exception:
        return []


def _build_all_colmap_cam_infos(dataset) -> List:
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
        return list(info.train_cameras)
    except Exception:
        return []


def _build_all_colmap_cameras(dataset) -> List:
    all_infos = _build_all_colmap_cam_infos(dataset)
    if len(all_infos) == 0:
        return []
    try:
        return cameraList_from_camInfos(all_infos, 1.0, dataset)
    except Exception:
        return []


def build_prism_validation_views(
    scene,
    dataset,
    cfg: PrismValidationConfig,
    proxy_ctx: Optional[GeometryProxyContext] = None,
    proxy_cfg: Optional[GeometryProxyConfig] = None,
) -> List:
    """
    Dev validation set for PRISM (hybrid):
    - file split: dropped buffer views first
    - then add train views prioritized by sparse observability
    """
    max_views = max(1, int(cfg.max_views))
    selected: List = []
    selected_names = set()
    proxy_cfg = proxy_cfg or GeometryProxyConfig(compute_normal=True)

    use_file_split = str(getattr(dataset, "split_strategy", "")).strip().lower() == "file"
    dropped = _load_split_dropped_keys(str(getattr(dataset, "split_file", ""))) if use_file_split else []
    if len(dropped) > 0:
        all_cams = _build_all_colmap_cameras(dataset)
        by_name = {normalize_image_key(getattr(c, "image_name", "")): c for c in all_cams}
        for key in dropped:
            cam = by_name.get(key, None)
            if cam is None:
                continue
            selected.append(cam)
            selected_names.add(key)
            if len(selected) >= int(min(max_views, max(0, cfg.num_buffer_views))):
                break

    train_pool = scene.getTrainCameras()
    pool_size = min(int(cfg.train_pool_size), len(train_pool))
    pool = train_pool[:pool_size]

    ranked = []
    for cam in pool:
        name = normalize_image_key(getattr(cam, "image_name", ""))
        if name in selected_names:
            continue
        score = 0.0
        depth_matches = 0
        normal_matches = 0
        if proxy_ctx is not None:
            obs = estimate_view_sparse_observability(view=cam, ctx=proxy_ctx, cfg=proxy_cfg)
            depth_matches = int(obs["depth_matches"])
            normal_matches = int(obs["normal_matches"])
            score = float(obs["score"])
            if bool(cfg.prefer_observable_train_views):
                if depth_matches < int(max(0, cfg.min_depth_matches_per_view)):
                    continue
                if bool(proxy_cfg.compute_normal) and normal_matches < int(max(0, cfg.min_normal_matches_per_view)):
                    continue
        ranked.append((score, depth_matches, normal_matches, cam))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    for _, _, _, cam in ranked:
        name = normalize_image_key(getattr(cam, "image_name", ""))
        if name in selected_names:
            continue
        selected.append(cam)
        selected_names.add(name)
        if len(selected) >= int(min(max_views, max(0, cfg.num_buffer_views) + max(0, cfg.num_train_views))):
            break

    if len(selected) == 0:
        for cam in scene.getTestCameras():
            selected.append(cam)
            if len(selected) >= max_views:
                break
    if len(selected) > max_views:
        selected = selected[:max_views]
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
    proxy_ctx: GeometryProxyContext,
    proxy_cfg: GeometryProxyConfig,
    cfg: PrismValidationConfig,
) -> Dict[str, float]:
    psnr_vals = []
    mae_vals = []
    absrel_num = 0.0
    delta_num = 0.0
    depth_weight = 0.0
    mean_angle_num = 0.0
    abs_cos_num = 0.0
    normal_weight = 0.0
    num_depth_views = 0
    num_normal_views = 0
    total_depth_matches = 0
    total_normal_matches = 0
    dropped_reason_breakdown: Dict[str, int] = {}

    roi_psnr_vals = []
    roi_mae_vals = []

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

            proxy = evaluate_view_sparse_geometry_proxy(
                view=v,
                render_pkg=pkg,
                ctx=proxy_ctx,
                cfg=proxy_cfg,
            )
            reason = str(proxy.get("reason", "unknown"))
            if reason != "ok":
                dropped_reason_breakdown[reason] = int(dropped_reason_breakdown.get(reason, 0)) + 1

            depth_points = int(proxy.get("depth_points", 0))
            if depth_points > 0 and proxy.get("depth_stats", None) is not None:
                ds = proxy["depth_stats"]
                num_depth_views += 1
                total_depth_matches += depth_points
                depth_weight += float(depth_points)
                absrel_num += float(ds["abs_rel"]) * float(depth_points)
                delta_num += float(ds["delta_1.25"]) * float(depth_points)

            normal_points = int(proxy.get("normal_points", 0))
            if bool(proxy_cfg.compute_normal) and normal_points > 0 and proxy.get("normal_stats", None) is not None:
                ns = proxy["normal_stats"]
                num_normal_views += 1
                total_normal_matches += normal_points
                normal_weight += float(normal_points)
                mean_angle_num += float(ns["mean_ang_deg"]) * float(normal_points)
                abs_cos_num += float(ns["mean_abs_cos"]) * float(normal_points)

            gm = getattr(v, "ground_mask", None)
            if gm is not None:
                mask = gm.to(img.device)
                if mask.ndim == 3:
                    mask = mask[0]
                if mask.ndim == 2:
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

    geometry_failure_reasons: List[str] = []
    if total_depth_matches <= 0:
        if int(dropped_reason_breakdown.get("render_missing_output", 0)) > 0:
            geometry_failure_reasons.append("render_missing_output")
        else:
            geometry_failure_reasons.append("no_sparse_matches")
    if total_depth_matches < int(max(0, cfg.min_valid_depth_matches)):
        geometry_failure_reasons.append("insufficient_depth_matches")
    if bool(proxy_cfg.compute_normal):
        if total_normal_matches <= 0:
            geometry_failure_reasons.append("no_sparse_matches")
        if total_normal_matches < int(max(0, cfg.min_valid_normal_matches)):
            geometry_failure_reasons.append("insufficient_normal_matches")

    geometry_observable = len(geometry_failure_reasons) == 0

    return {
        "num_views": float(len(views)),
        "num_depth_views_used": float(num_depth_views),
        "num_normal_views_used": float(num_normal_views),
        "total_valid_depth_matches": float(total_depth_matches),
        "total_valid_normal_matches": float(total_normal_matches),
        "dropped_views_reason_breakdown": {str(k): int(v) for k, v in dropped_reason_breakdown.items()},
        "geometry_observable": 1.0 if geometry_observable else 0.0,
        "geometry_failure_reasons": list(sorted(set(geometry_failure_reasons))),
        "psnr": _safe_mean(psnr_vals),
        "mae": _safe_mean(mae_vals),
        "absrel": float(absrel_num / depth_weight) if depth_weight > 0 else float("nan"),
        "delta_1.25": float(delta_num / depth_weight) if depth_weight > 0 else float("nan"),
        "mean_angle": float(mean_angle_num / normal_weight) if normal_weight > 0 else float("nan"),
        "abs_cos": float(abs_cos_num / normal_weight) if normal_weight > 0 else float("nan"),
        "roi_psnr": _safe_mean(roi_psnr_vals),
        "roi_mae": _safe_mean(roi_mae_vals),
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
    rules: List[str] = []

    cur_observable = bool(float(current_metrics.get("geometry_observable", 0.0)) > 0.5)
    if not cur_observable:
        reasons = current_metrics.get("geometry_failure_reasons", [])
        if isinstance(reasons, list) and len(reasons) > 0:
            rules.extend([str(r) for r in reasons])
        else:
            rules.append("insufficient_geometry_observability")
        return False, deltas, sorted(set(rules))

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
