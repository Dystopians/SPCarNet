import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.prism_geometry_proxy import (
    GeometryProxyConfig,
    GeometryProxyContext,
    estimate_view_sparse_observability,
    evaluate_view_sparse_geometry_proxy,
    normalize_image_key,
)
from utils.prism_scoring import PrismScoreOutputs


@dataclass
class CounterfactualGateConfig:
    min_delta_psnr_db: float = -0.05
    max_delta_mae: float = 0.002
    max_delta_absrel: float = 0.0008
    max_baseline_absrel_for_absrel_check: float = float("inf")
    max_delta_mean_angle_deg: float = 0.3
    max_changed_pixel_ratio: float = 0.005
    changed_pixel_threshold: float = 0.02
    min_valid_depth_matches: int = 128
    min_valid_normal_matches: int = 64


@dataclass
class CalibrationConfig:
    num_buffer_views: int = 8
    num_hard_train_views: int = 8
    hard_view_pool_size: int = 64
    prefer_observable_views: bool = True
    min_depth_matches_per_view: int = 24
    min_normal_matches_per_view: int = 8
    diverse_views: bool = False
    num_diverse_test_views: int = 0
    num_diverse_train_views: int = 0


@dataclass
class CompactionSelectionConfig:
    microbatch_active_ratio: float = 0.0035
    candidate_pool_multiplier: float = 6.0
    min_prune_count: int = 256
    roi_budget_fraction: float = 0.10
    near_field_budget_fraction: float = 0.25
    roi_signal_threshold: float = 0.05
    near_field_area_percentile: float = 80.0


@dataclass
class CounterfactualDecision:
    accept: bool
    num_candidates: int
    deltas: Dict[str, Any]
    baseline: Dict[str, Any]
    counterfactual: Dict[str, Any]
    reason: str
    num_views: int
    num_depth_views_used: int
    num_normal_views_used: int
    total_valid_depth_matches: int
    total_valid_normal_matches: int
    dropped_views_reason_breakdown: Dict[str, int]


class TemporaryTriangleMask:
    """
    RAII-style temporary active mask with guaranteed rollback.
    """

    def __init__(self, triangles, inactive_triangle_ids: torch.Tensor):
        self.triangles = triangles
        self.inactive_triangle_ids = inactive_triangle_ids.to(torch.int64)
        self._prev_mask = None

    def __enter__(self):
        t = int(self.triangles._triangle_indices.shape[0])
        if self._prev_mask is None:
            self._prev_mask = self.triangles.get_temporary_active_mask()
        if self._prev_mask is None:
            active = torch.ones((t,), dtype=torch.bool, device=self.triangles._triangle_indices.device)
        else:
            active = self._prev_mask.clone()
        ids = self.inactive_triangle_ids
        valid = (ids >= 0) & (ids < t)
        if torch.any(valid):
            active[ids[valid]] = False
        self.triangles.set_temporary_active_mask(active)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._prev_mask is None:
            self.triangles.clear_temporary_active_mask()
        else:
            self.triangles.set_temporary_active_mask(self._prev_mask)
        return False


def _load_split_dropped(split_file: str) -> List[str]:
    if not split_file or (not os.path.exists(split_file)):
        return []
    try:
        with open(split_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        dropped = payload.get("dropped", [])
        return [normalize_image_key(x) for x in dropped]
    except Exception:
        return []


def _build_all_colmap_cam_infos(dataset) -> List:
    try:
        if not os.path.exists(os.path.join(dataset.source_path, "sparse")):
            return []
        all_info = sceneLoadTypeCallbacks["Colmap"](
            dataset.source_path,
            dataset.images,
            False,
            split_strategy="llff",
            split_file="",
        )
        return list(all_info.train_cameras)
    except Exception:
        return []


def _build_all_cameras(scene, dataset) -> List:
    all_infos = _build_all_colmap_cam_infos(dataset)
    if len(all_infos) == 0:
        return []
    try:
        return cameraList_from_camInfos(all_infos, 1.0, dataset)
    except Exception:
        return []


def _compute_view_difficulty(
    render_pkg: Dict,
    view,
    proxy_ctx: GeometryProxyContext,
    proxy_cfg: GeometryProxyConfig,
) -> float:
    img = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
    gt = torch.clamp(view.original_image.to(img.device), 0.0, 1.0)
    mae = float(torch.mean(torch.abs(img - gt)).item())

    proxy = evaluate_view_sparse_geometry_proxy(
        view=view,
        render_pkg=render_pkg,
        ctx=proxy_ctx,
        cfg=proxy_cfg,
    )
    depth_penalty = 1.0
    normal_penalty = 0.5 if bool(proxy_cfg.compute_normal) else 0.0
    if proxy.get("depth_stats", None) is not None:
        depth_penalty = float(proxy["depth_stats"]["abs_rel"])
    if bool(proxy_cfg.compute_normal) and proxy.get("normal_stats", None) is not None:
        normal_penalty = float(proxy["normal_stats"]["mean_ang_deg"]) / 180.0
    return float(mae + depth_penalty + normal_penalty)


def _observability_passes(obs: Dict[str, Any], proxy_cfg: GeometryProxyConfig, cfg: CalibrationConfig) -> bool:
    if int(obs["depth_matches"]) < int(max(0, cfg.min_depth_matches_per_view)):
        return False
    if bool(proxy_cfg.compute_normal):
        if int(obs["normal_matches"]) < int(max(0, cfg.min_normal_matches_per_view)):
            return False
    return True


def _is_view_observable(
    view,
    proxy_ctx: GeometryProxyContext,
    proxy_cfg: GeometryProxyConfig,
    cfg: CalibrationConfig,
) -> bool:
    obs = estimate_view_sparse_observability(view=view, ctx=proxy_ctx, cfg=proxy_cfg)
    return _observability_passes(obs, proxy_cfg, cfg)


def _evenly_spaced(items: Sequence, count: int) -> List:
    if count <= 0 or len(items) == 0:
        return []
    n = min(int(count), len(items))
    if n == len(items):
        return list(items)
    idxs = np.linspace(0, len(items) - 1, num=n)
    out = []
    used = set()
    for idx in idxs:
        i = int(round(float(idx)))
        if i in used:
            continue
        used.add(i)
        out.append(items[i])
    return out


def _try_append_calibration_view(
    selected: List,
    selected_names: set,
    cam,
    source: str,
    proxy_ctx: GeometryProxyContext,
    proxy_cfg: GeometryProxyConfig,
    cfg: CalibrationConfig,
    manifest: Optional[List[Dict]] = None,
) -> bool:
    name = normalize_image_key(getattr(cam, "image_name", ""))
    if not name or name in selected_names:
        return False
    obs = estimate_view_sparse_observability(view=cam, ctx=proxy_ctx, cfg=proxy_cfg)
    if bool(cfg.prefer_observable_views) and not _observability_passes(obs, proxy_cfg, cfg):
        return False
    selected.append(cam)
    selected_names.add(name)
    if manifest is not None:
        manifest.append(
            {
                "image_name": str(getattr(cam, "image_name", "")),
                "normalized_name": str(name),
                "source": str(source),
                "depth_matches": int(obs.get("depth_matches", 0)),
                "normal_matches": int(obs.get("normal_matches", 0)),
                "observability_score": float(obs.get("score", 0.0)),
                "observability_reason": str(obs.get("reason", "unknown")),
            }
        )
    return True


def build_calibration_set(
    scene,
    dataset,
    triangles,
    render_func,
    pipe,
    background: torch.Tensor,
    cfg: CalibrationConfig,
    proxy_ctx: GeometryProxyContext,
    proxy_cfg: GeometryProxyConfig,
) -> List:
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    selected: List = []
    selected_names = set()
    manifest: List[Dict] = []

    if bool(cfg.diverse_views):
        n_test = int(cfg.num_diverse_test_views) if int(cfg.num_diverse_test_views) > 0 else int(cfg.num_buffer_views)
        for cam in _evenly_spaced(test_cams, n_test):
            _try_append_calibration_view(
                selected=selected,
                selected_names=selected_names,
                cam=cam,
                source="diverse_test",
                proxy_ctx=proxy_ctx,
                proxy_cfg=proxy_cfg,
                cfg=cfg,
                manifest=manifest,
            )
        train_pool = train_cams[: min(len(train_cams), int(cfg.hard_view_pool_size))]
        n_train = int(cfg.num_diverse_train_views) if int(cfg.num_diverse_train_views) > 0 else int(cfg.num_hard_train_views)
        for cam in _evenly_spaced(train_pool, n_train):
            _try_append_calibration_view(
                selected=selected,
                selected_names=selected_names,
                cam=cam,
                source="diverse_train",
                proxy_ctx=proxy_ctx,
                proxy_cfg=proxy_cfg,
                cfg=cfg,
                manifest=manifest,
            )

    dropped = []
    if getattr(dataset, "split_strategy", "") == "file":
        dropped = _load_split_dropped(getattr(dataset, "split_file", ""))
    if len(dropped) > 0:
        by_name = {}
        for cam in test_cams + train_cams:
            by_name[normalize_image_key(getattr(cam, "image_name", ""))] = cam
        for name in dropped:
            cam = by_name.get(name, None)
            if cam is not None and name not in selected_names:
                if (not bool(cfg.prefer_observable_views)) or _is_view_observable(cam, proxy_ctx, proxy_cfg, cfg):
                    selected.append(cam)
                    selected_names.add(name)
                    manifest.append({"image_name": str(getattr(cam, "image_name", "")), "normalized_name": str(name), "source": "split_dropped"})
                if len(selected) >= int(cfg.num_buffer_views):
                    break
        if len(selected) < int(cfg.num_buffer_views):
            all_cams = _build_all_cameras(scene=scene, dataset=dataset)
            by_name_all = {normalize_image_key(getattr(c, "image_name", "")): c for c in all_cams}
            for name in dropped:
                if name in selected_names:
                    continue
                cam = by_name_all.get(name, None)
                if cam is None:
                    continue
                if (not bool(cfg.prefer_observable_views)) or _is_view_observable(cam, proxy_ctx, proxy_cfg, cfg):
                    selected.append(cam)
                    selected_names.add(name)
                    manifest.append({"image_name": str(getattr(cam, "image_name", "")), "normalized_name": str(name), "source": "split_dropped_all"})
                if len(selected) >= int(cfg.num_buffer_views):
                    break

    if len(selected) < int(cfg.num_buffer_views):
        for cam in test_cams:
            name = normalize_image_key(getattr(cam, "image_name", ""))
            if name in selected_names:
                continue
            if (not bool(cfg.prefer_observable_views)) or _is_view_observable(cam, proxy_ctx, proxy_cfg, cfg):
                selected.append(cam)
                selected_names.add(name)
                obs = estimate_view_sparse_observability(view=cam, ctx=proxy_ctx, cfg=proxy_cfg)
                manifest.append(
                    {
                        "image_name": str(getattr(cam, "image_name", "")),
                        "normalized_name": str(name),
                        "source": "test_prefix",
                        "depth_matches": int(obs.get("depth_matches", 0)),
                        "normal_matches": int(obs.get("normal_matches", 0)),
                        "observability_score": float(obs.get("score", 0.0)),
                        "observability_reason": str(obs.get("reason", "unknown")),
                    }
                )
            if len(selected) >= int(cfg.num_buffer_views):
                break

    pool_size = min(len(train_cams), int(cfg.hard_view_pool_size))
    pool = train_cams[:pool_size]
    filtered = []
    for cam in pool:
        name = normalize_image_key(getattr(cam, "image_name", ""))
        if name in selected_names:
            continue
        if (not bool(cfg.prefer_observable_views)) or _is_view_observable(cam, proxy_ctx, proxy_cfg, cfg):
            filtered.append(cam)
    if len(filtered) == 0:
        filtered = [c for c in pool if normalize_image_key(getattr(c, "image_name", "")) not in selected_names]

    hardness = []
    with torch.no_grad():
        for cam in filtered:
            pkg = render_func(cam, triangles, pipe, background)
            h = _compute_view_difficulty(
                render_pkg=pkg,
                view=cam,
                proxy_ctx=proxy_ctx,
                proxy_cfg=proxy_cfg,
            )
            hardness.append(h)
    if len(hardness) > 0:
        order = np.argsort(np.asarray(hardness))[::-1]
        n_hard = int(cfg.num_hard_train_views)
        for idx in order[:n_hard]:
            cam = filtered[int(idx)]
            name = normalize_image_key(getattr(cam, "image_name", ""))
            if name in selected_names:
                continue
            selected.append(cam)
            selected_names.add(name)
            obs = estimate_view_sparse_observability(view=cam, ctx=proxy_ctx, cfg=proxy_cfg)
            manifest.append(
                {
                    "image_name": str(getattr(cam, "image_name", "")),
                    "normalized_name": str(name),
                    "source": "hard_train",
                    "hardness": float(hardness[int(idx)]),
                    "depth_matches": int(obs.get("depth_matches", 0)),
                    "normal_matches": int(obs.get("normal_matches", 0)),
                    "observability_score": float(obs.get("score", 0.0)),
                    "observability_reason": str(obs.get("reason", "unknown")),
                }
            )
    for i, cam in enumerate(selected):
        try:
            setattr(cam, "_prism_calibration_index", int(i))
        except Exception:
            pass
    try:
        setattr(cfg, "_last_manifest", manifest)
    except Exception:
        pass
    return selected


def _compute_image_metrics(render_pkg: Dict, view) -> Dict[str, float]:
    img = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
    gt = torch.clamp(view.original_image.to(img.device), 0.0, 1.0)
    mae = float(torch.mean(torch.abs(img - gt)).item())
    mse = float(torch.mean((img - gt) ** 2).item())
    psnr = float(-10.0 * np.log10(max(mse, 1e-8)))
    return {"psnr": psnr, "mae": mae, "image": img}


def _aggregate_from_entries(entries: Sequence[Dict], compute_normal: bool) -> Dict[str, float]:
    psnr_vals = [float(e["psnr"]) for e in entries if np.isfinite(e["psnr"])]
    mae_vals = [float(e["mae"]) for e in entries if np.isfinite(e["mae"])]

    depth_weight = 0.0
    absrel_num = 0.0
    d125_num = 0.0
    dmae_num = 0.0
    normal_weight = 0.0
    mean_ang_num = 0.0
    abs_cos_num = 0.0
    reason_breakdown: Dict[str, int] = {}
    num_depth_views = 0
    num_normal_views = 0
    total_depth_matches = 0
    total_normal_matches = 0

    for e in entries:
        proxy = e["proxy"]
        reason = str(proxy.get("reason", "unknown"))
        if reason != "ok":
            reason_breakdown[reason] = int(reason_breakdown.get(reason, 0)) + 1
        depth_points = int(proxy.get("depth_points", 0))
        normal_points = int(proxy.get("normal_points", 0))
        if depth_points > 0 and proxy.get("depth_stats", None) is not None:
            ds = proxy["depth_stats"]
            num_depth_views += 1
            total_depth_matches += depth_points
            depth_weight += float(depth_points)
            absrel_num += float(ds["abs_rel"]) * float(depth_points)
            d125_num += float(ds["delta_1.25"]) * float(depth_points)
            dmae_num += float(ds["mae"]) * float(depth_points)
        if compute_normal and normal_points > 0 and proxy.get("normal_stats", None) is not None:
            ns = proxy["normal_stats"]
            num_normal_views += 1
            total_normal_matches += normal_points
            normal_weight += float(normal_points)
            mean_ang_num += float(ns["mean_ang_deg"]) * float(normal_points)
            abs_cos_num += float(ns["mean_abs_cos"]) * float(normal_points)

    return {
        "psnr": float(np.mean(psnr_vals)) if len(psnr_vals) > 0 else float("nan"),
        "mae": float(np.mean(mae_vals)) if len(mae_vals) > 0 else float("nan"),
        "absrel": float(absrel_num / depth_weight) if depth_weight > 0 else float("nan"),
        "delta_125": float(d125_num / depth_weight) if depth_weight > 0 else float("nan"),
        "depth_mae": float(dmae_num / depth_weight) if depth_weight > 0 else float("nan"),
        "mean_normal_angle": float(mean_ang_num / normal_weight) if normal_weight > 0 else float("nan"),
        "abs_cos": float(abs_cos_num / normal_weight) if normal_weight > 0 else float("nan"),
        "num_depth_views_used": int(num_depth_views),
        "num_normal_views_used": int(num_normal_views),
        "total_valid_depth_matches": int(total_depth_matches),
        "total_valid_normal_matches": int(total_normal_matches),
        "dropped_views_reason_breakdown": reason_breakdown,
    }


def run_counterfactual_simulation(
    scene,
    triangles,
    render_func,
    pipe,
    background: torch.Tensor,
    candidate_triangle_ids: torch.Tensor,
    calibration_views: Sequence,
    gate_cfg: CounterfactualGateConfig,
    proxy_ctx: GeometryProxyContext,
    proxy_cfg: GeometryProxyConfig,
) -> CounterfactualDecision:
    cand = candidate_triangle_ids.to(torch.int64)
    t = int(triangles._triangle_indices.shape[0])
    valid = (cand >= 0) & (cand < t)
    cand = cand[valid]
    if cand.numel() == 0:
        return CounterfactualDecision(
            accept=False,
            num_candidates=0,
            deltas={},
            baseline={},
            counterfactual={},
            reason="empty_candidates",
            num_views=0,
            num_depth_views_used=0,
            num_normal_views_used=0,
            total_valid_depth_matches=0,
            total_valid_normal_matches=0,
            dropped_views_reason_breakdown={},
        )
    if len(calibration_views) == 0:
        return CounterfactualDecision(
            accept=False,
            num_candidates=int(cand.numel()),
            deltas={},
            baseline={},
            counterfactual={},
            reason="empty_calibration_set",
            num_views=0,
            num_depth_views_used=0,
            num_normal_views_used=0,
            total_valid_depth_matches=0,
            total_valid_normal_matches=0,
            dropped_views_reason_breakdown={},
        )

    baseline_entries = []
    with torch.no_grad():
        for v in calibration_views:
            pkg = render_func(v, triangles, pipe, background)
            img_m = _compute_image_metrics(pkg, v)
            proxy_m = evaluate_view_sparse_geometry_proxy(v, pkg, proxy_ctx, proxy_cfg)
            baseline_entries.append({"psnr": img_m["psnr"], "mae": img_m["mae"], "image": img_m["image"], "proxy": proxy_m})

    cf_entries = []
    with TemporaryTriangleMask(triangles=triangles, inactive_triangle_ids=cand):
        with torch.no_grad():
            for i, v in enumerate(calibration_views):
                pkg = render_func(v, triangles, pipe, background)
                img_m = _compute_image_metrics(pkg, v)
                proxy_m = evaluate_view_sparse_geometry_proxy(v, pkg, proxy_ctx, proxy_cfg)
                base_img = baseline_entries[i]["image"]
                cf_img = img_m["image"]
                diff = torch.mean(torch.abs(base_img - cf_img), dim=0)
                changed = torch.mean((diff > float(gate_cfg.changed_pixel_threshold)).to(torch.float32))
                cf_entries.append(
                    {
                        "psnr": img_m["psnr"],
                        "mae": img_m["mae"],
                        "changed_pixel_ratio": float(changed.item()),
                        "proxy": proxy_m,
                    }
                )

    baseline = _aggregate_from_entries(entries=baseline_entries, compute_normal=bool(proxy_cfg.compute_normal))
    counterfactual = _aggregate_from_entries(entries=cf_entries, compute_normal=bool(proxy_cfg.compute_normal))
    changed_vals = [float(e["changed_pixel_ratio"]) for e in cf_entries if np.isfinite(e["changed_pixel_ratio"])]
    counterfactual["changed_pixel_ratio"] = float(np.mean(changed_vals)) if len(changed_vals) > 0 else float("nan")
    per_view_deltas = []
    for i, v in enumerate(calibration_views):
        b = baseline_entries[i]
        c = cf_entries[i]
        b_proxy = b.get("proxy", {})
        c_proxy = c.get("proxy", {})
        b_depth = b_proxy.get("depth_stats", None)
        c_depth = c_proxy.get("depth_stats", None)
        b_norm = b_proxy.get("normal_stats", None)
        c_norm = c_proxy.get("normal_stats", None)
        per_view_deltas.append(
            {
                "index": int(i),
                "image_name": str(getattr(v, "image_name", "")),
                "delta_psnr": float(c["psnr"] - b["psnr"]) if np.isfinite(c["psnr"]) and np.isfinite(b["psnr"]) else float("nan"),
                "delta_mae": float(c["mae"] - b["mae"]) if np.isfinite(c["mae"]) and np.isfinite(b["mae"]) else float("nan"),
                "changed_pixel_ratio": float(c.get("changed_pixel_ratio", float("nan"))),
                "baseline_depth_points": int(b_proxy.get("depth_points", 0)),
                "counterfactual_depth_points": int(c_proxy.get("depth_points", 0)),
                "delta_absrel": float(c_depth["abs_rel"] - b_depth["abs_rel"]) if b_depth is not None and c_depth is not None else float("nan"),
                "delta_mean_angle": float(c_norm["mean_ang_deg"] - b_norm["mean_ang_deg"]) if b_norm is not None and c_norm is not None else float("nan"),
                "baseline_reason": str(b_proxy.get("reason", "unknown")),
                "counterfactual_reason": str(c_proxy.get("reason", "unknown")),
            }
        )
    counterfactual["per_view_deltas"] = per_view_deltas

    combined_reasons: Dict[str, int] = {}
    for reason_map in [baseline["dropped_views_reason_breakdown"], counterfactual["dropped_views_reason_breakdown"]]:
        for k, v in reason_map.items():
            combined_reasons[k] = int(combined_reasons.get(k, 0)) + int(v)

    min_depth = int(max(0, gate_cfg.min_valid_depth_matches))
    min_normal = int(max(0, gate_cfg.min_valid_normal_matches)) if bool(proxy_cfg.compute_normal) else 0
    b_depth = int(baseline["total_valid_depth_matches"])
    c_depth = int(counterfactual["total_valid_depth_matches"])
    b_norm = int(baseline["total_valid_normal_matches"])
    c_norm = int(counterfactual["total_valid_normal_matches"])

    if max(b_depth, c_depth) <= 0:
        reason = "render_missing_output" if int(combined_reasons.get("render_missing_output", 0)) > 0 else "no_sparse_matches"
        return CounterfactualDecision(
            accept=False,
            num_candidates=int(cand.numel()),
            deltas={},
            baseline=baseline,
            counterfactual=counterfactual,
            reason=reason,
            num_views=int(len(calibration_views)),
            num_depth_views_used=int(baseline["num_depth_views_used"]),
            num_normal_views_used=int(baseline["num_normal_views_used"]),
            total_valid_depth_matches=int(b_depth),
            total_valid_normal_matches=int(b_norm),
            dropped_views_reason_breakdown=combined_reasons,
        )
    if (b_depth < min_depth) or (c_depth < min_depth):
        return CounterfactualDecision(
            accept=False,
            num_candidates=int(cand.numel()),
            deltas={},
            baseline=baseline,
            counterfactual=counterfactual,
            reason="insufficient_depth_matches",
            num_views=int(len(calibration_views)),
            num_depth_views_used=int(baseline["num_depth_views_used"]),
            num_normal_views_used=int(baseline["num_normal_views_used"]),
            total_valid_depth_matches=int(b_depth),
            total_valid_normal_matches=int(b_norm),
            dropped_views_reason_breakdown=combined_reasons,
        )
    if min_normal > 0 and ((b_norm < min_normal) or (c_norm < min_normal)):
        return CounterfactualDecision(
            accept=False,
            num_candidates=int(cand.numel()),
            deltas={},
            baseline=baseline,
            counterfactual=counterfactual,
            reason="insufficient_normal_matches",
            num_views=int(len(calibration_views)),
            num_depth_views_used=int(baseline["num_depth_views_used"]),
            num_normal_views_used=int(baseline["num_normal_views_used"]),
            total_valid_depth_matches=int(b_depth),
            total_valid_normal_matches=int(b_norm),
            dropped_views_reason_breakdown=combined_reasons,
        )

    deltas = {
        "delta_psnr": counterfactual["psnr"] - baseline["psnr"]
        if np.isfinite(counterfactual["psnr"]) and np.isfinite(baseline["psnr"])
        else float("nan"),
        "delta_mae": counterfactual["mae"] - baseline["mae"]
        if np.isfinite(counterfactual["mae"]) and np.isfinite(baseline["mae"])
        else float("nan"),
        "delta_absrel": counterfactual["absrel"] - baseline["absrel"]
        if np.isfinite(counterfactual["absrel"]) and np.isfinite(baseline["absrel"])
        else float("nan"),
        "delta_mean_angle": counterfactual["mean_normal_angle"] - baseline["mean_normal_angle"]
        if np.isfinite(counterfactual["mean_normal_angle"]) and np.isfinite(baseline["mean_normal_angle"])
        else float("nan"),
        "changed_pixel_ratio": counterfactual["changed_pixel_ratio"],
    }

    absrel_check_reliable = True
    if np.isfinite(baseline["absrel"]):
        absrel_check_reliable = baseline["absrel"] <= float(gate_cfg.max_baseline_absrel_for_absrel_check)

    checks = []
    if np.isfinite(deltas["delta_psnr"]):
        checks.append(deltas["delta_psnr"] >= float(gate_cfg.min_delta_psnr_db))
    if np.isfinite(deltas["delta_mae"]):
        checks.append(deltas["delta_mae"] <= float(gate_cfg.max_delta_mae))
    if np.isfinite(deltas["delta_absrel"]) and bool(absrel_check_reliable):
        checks.append(deltas["delta_absrel"] <= float(gate_cfg.max_delta_absrel))
    if np.isfinite(deltas["delta_mean_angle"]):
        checks.append(deltas["delta_mean_angle"] <= float(gate_cfg.max_delta_mean_angle_deg))
    if np.isfinite(deltas["changed_pixel_ratio"]):
        checks.append(deltas["changed_pixel_ratio"] <= float(gate_cfg.max_changed_pixel_ratio))

    deltas["absrel_check_reliable"] = bool(absrel_check_reliable)
    deltas["max_baseline_absrel_for_absrel_check"] = float(gate_cfg.max_baseline_absrel_for_absrel_check)

    accept = bool(all(checks)) if len(checks) > 0 else False
    return CounterfactualDecision(
        accept=accept,
        num_candidates=int(cand.numel()),
        deltas=deltas,
        baseline=baseline,
        counterfactual=counterfactual,
        reason="ok" if accept else "threshold_reject",
        num_views=int(len(calibration_views)),
        num_depth_views_used=int(baseline["num_depth_views_used"]),
        num_normal_views_used=int(baseline["num_normal_views_used"]),
        total_valid_depth_matches=int(b_depth),
        total_valid_normal_matches=int(b_norm),
        dropped_views_reason_breakdown=combined_reasons,
    )


def counterfactual_decision_to_dict(decision: CounterfactualDecision) -> Dict:
    payload = asdict(decision)
    payload["accept"] = bool(payload["accept"])
    payload["num_candidates"] = int(payload["num_candidates"])
    payload["num_views"] = int(payload["num_views"])
    payload["num_depth_views_used"] = int(payload["num_depth_views_used"])
    payload["num_normal_views_used"] = int(payload["num_normal_views_used"])
    payload["total_valid_depth_matches"] = int(payload["total_valid_depth_matches"])
    payload["total_valid_normal_matches"] = int(payload["total_valid_normal_matches"])
    payload["dropped_views_reason_breakdown"] = {
        str(k): int(v) for k, v in payload.get("dropped_views_reason_breakdown", {}).items()
    }
    return payload


def select_prism_candidate_ids(
    scores: PrismScoreOutputs,
    dead_prune_ratio: float,
    candidate_prune_ratio: float,
    candidate_max_count: int = 0,
    rank_score_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    t = int(scores.prune_score_t.numel())
    if t == 0:
        return torch.zeros((0,), dtype=torch.int64, device=scores.prune_score_t.device)
    rank_scores = scores.prune_score_t
    if rank_score_t is not None and int(rank_score_t.numel()) == t:
        rank_scores = rank_score_t.to(device=scores.prune_score_t.device, dtype=torch.float32)

    dead_ids = torch.nonzero(scores.dead_mask, as_tuple=True)[0]
    cand_ids = torch.nonzero(scores.candidate_mask, as_tuple=True)[0]

    dead_n = int(max(0, dead_prune_ratio) * t)
    cand_n = int(max(0, candidate_prune_ratio) * t)
    if candidate_max_count > 0:
        cand_n = min(cand_n, int(candidate_max_count))

    selected = []
    if dead_ids.numel() > 0 and dead_n > 0:
        s = rank_scores[dead_ids]
        k = min(dead_n, int(dead_ids.numel()))
        _, idx = torch.topk(s, k=k, largest=True, sorted=False)
        selected.append(dead_ids[idx])
    if cand_ids.numel() > 0 and cand_n > 0:
        s = rank_scores[cand_ids]
        k = min(cand_n, int(cand_ids.numel()))
        _, idx = torch.topk(s, k=k, largest=True, sorted=False)
        selected.append(cand_ids[idx])

    if len(selected) == 0:
        return torch.zeros((0,), dtype=torch.int64, device=scores.prune_score_t.device)
    return torch.unique(torch.cat(selected, dim=0))


def select_prism_compaction_microbatch_ids(
    scores: PrismScoreOutputs,
    projected_area_ema: torch.Tensor,
    cfg: CompactionSelectionConfig,
    rejected_mask: Optional[torch.Tensor] = None,
):
    t = int(scores.prune_score_t.numel())
    device = scores.prune_score_t.device
    empty = torch.zeros((0,), dtype=torch.int64, device=device)
    if t == 0:
        return empty, {"target_count": 0, "pool_count": 0, "roi_count": 0, "near_field_count": 0, "far_field_count": 0}

    candidate_mask = scores.candidate_mask.to(torch.bool)
    if rejected_mask is not None:
        candidate_mask = candidate_mask & (~rejected_mask.to(device=device, dtype=torch.bool))
    cand_ids = torch.nonzero(candidate_mask, as_tuple=True)[0]
    active_count = int(cand_ids.numel())
    if active_count <= 0:
        return empty, {"target_count": 0, "pool_count": 0, "roi_count": 0, "near_field_count": 0, "far_field_count": 0}

    target_count = int(max(int(getattr(cfg, "min_prune_count", 256)), round(float(getattr(cfg, "microbatch_active_ratio", 0.0035)) * active_count)))
    target_count = min(target_count, active_count)
    if target_count <= 0:
        return empty, {"target_count": 0, "pool_count": 0, "roi_count": 0, "near_field_count": 0, "far_field_count": 0}

    pool_count = int(max(target_count, round(float(getattr(cfg, "candidate_pool_multiplier", 6.0)) * target_count)))
    pool_count = min(pool_count, active_count)
    cand_scores = scores.prune_score_t[cand_ids]
    _, top_idx = torch.topk(cand_scores, k=pool_count, largest=True, sorted=True)
    pool_ids = cand_ids[top_idx]

    roi_signal = scores.optional_roiprotect_t[pool_ids].to(torch.float32)
    area_vals = projected_area_ema[pool_ids].to(torch.float32)
    render_keep = scores.render_keep_t[pool_ids].to(torch.float32)
    if pool_ids.numel() > 1:
        q = float(max(0.0, min(1.0, float(getattr(cfg, "near_field_area_percentile", 80.0)) / 100.0)))
        near_thr = float(torch.quantile(area_vals, q=q).item())
    else:
        near_thr = float(area_vals.mean().item()) if pool_ids.numel() > 0 else 0.0
    roi_mask = roi_signal > float(getattr(cfg, "roi_signal_threshold", 0.05))
    near_mask = (~roi_mask) & ((area_vals >= near_thr) | (render_keep > float(getattr(cfg, "roi_signal_threshold", 0.05))))
    far_mask = (~roi_mask) & (~near_mask)

    def _pick(mask: torch.Tensor, k: int) -> torch.Tensor:
        ids = pool_ids[mask]
        if ids.numel() == 0 or k <= 0:
            return empty
        s = scores.prune_score_t[ids]
        kk = min(int(k), int(ids.numel()))
        _, idx = torch.topk(s, k=kk, largest=True, sorted=False)
        return ids[idx]

    roi_cap = int(round(target_count * float(getattr(cfg, "roi_budget_fraction", 0.10))))
    near_cap = int(round(target_count * float(getattr(cfg, "near_field_budget_fraction", 0.25))))
    far_cap = target_count

    selected_chunks = []
    far_sel = _pick(far_mask, far_cap)
    if far_sel.numel() > 0:
        selected_chunks.append(far_sel)
    remaining = max(0, target_count - sum(int(x.numel()) for x in selected_chunks))
    near_sel = _pick(near_mask, min(near_cap, remaining))
    if near_sel.numel() > 0:
        selected_chunks.append(near_sel)
    remaining = max(0, target_count - sum(int(x.numel()) for x in selected_chunks))
    roi_sel = _pick(roi_mask, min(roi_cap, remaining))
    if roi_sel.numel() > 0:
        selected_chunks.append(roi_sel)

    chosen = torch.unique(torch.cat(selected_chunks, dim=0)) if len(selected_chunks) > 0 else empty
    stats = {
        "target_count": int(target_count),
        "pool_count": int(pool_ids.numel()),
        "roi_count": int(roi_mask.sum().item()),
        "near_field_count": int(near_mask.sum().item()),
        "far_field_count": int(far_mask.sum().item()),
        "selected_count": int(chosen.numel()),
        "roi_cap": int(roi_cap),
        "near_field_cap": int(near_cap),
    }
    return chosen, stats
