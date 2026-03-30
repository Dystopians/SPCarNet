import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.geometry_metrics_utils import depth_metrics
from utils.prism_scoring import PrismScoreOutputs
from utils.triangle_stats import TriangleState


@dataclass
class CounterfactualGateConfig:
    min_delta_psnr_db: float = -0.05
    max_delta_mae: float = 0.002
    max_delta_absrel: float = 0.0008
    max_delta_mean_angle_deg: float = 0.3
    max_changed_pixel_ratio: float = 0.005
    changed_pixel_threshold: float = 0.02


@dataclass
class CalibrationConfig:
    num_buffer_views: int = 8
    num_hard_train_views: int = 8
    hard_view_pool_size: int = 64


@dataclass
class CounterfactualDecision:
    accept: bool
    num_candidates: int
    deltas: Dict[str, float]
    baseline: Dict[str, float]
    counterfactual: Dict[str, float]
    reason: str


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


def _normalize_name(name: str) -> str:
    base = os.path.basename(name)
    return os.path.splitext(base)[0].lower()


def _load_split_dropped(split_file: str) -> List[str]:
    if not split_file or (not os.path.exists(split_file)):
        return []
    try:
        with open(split_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        dropped = payload.get("dropped", [])
        return [_normalize_name(x) for x in dropped]
    except Exception:
        return []


def _ensure_all_cameras(scene, dataset) -> List:
    # Build all views without train/test split to recover dropped buffer views when possible.
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
        all_cams = cameraList_from_camInfos(all_info.train_cameras, 1.0, dataset)
        return all_cams
    except Exception:
        return []


def _compute_view_difficulty(render_pkg: Dict, view) -> float:
    img = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
    gt = torch.clamp(view.original_image.to(img.device), 0.0, 1.0)
    mae = torch.mean(torch.abs(img - gt))

    depth_absrel = torch.tensor(0.0, device=img.device)
    if getattr(view, "invdepthmap", None) is not None:
        pred_depth = render_pkg["surf_depth"].detach()[0]
        gt_inv = view.invdepthmap.to(pred_depth.device)[0]
        gt_depth = 1.0 / torch.clamp(gt_inv, min=1e-6)
        valid = torch.isfinite(gt_depth) & torch.isfinite(pred_depth) & (gt_depth > 1e-6) & (pred_depth > 1e-6)
        if torch.any(valid):
            pd = pred_depth[valid]
            gd = gt_depth[valid]
            depth_absrel = torch.mean(torch.abs(pd - gd) / torch.clamp(gd, min=1e-6))

    normal_term = torch.tensor(0.0, device=img.device)
    if getattr(view, "normal_map", None) is not None:
        rn = F.normalize(render_pkg["rend_normal"].detach(), dim=0, eps=1e-6)
        gn = F.normalize(view.normal_map.to(rn.device), dim=0, eps=1e-6)
        c = torch.clamp(torch.abs(torch.sum(rn * gn, dim=0)), 0.0, 1.0)
        ang = torch.rad2deg(torch.arccos(c))
        normal_term = torch.mean(ang) / 180.0

    return float((mae + depth_absrel + normal_term).item())


def build_calibration_set(
    scene,
    dataset,
    triangles,
    render_func,
    pipe,
    background: torch.Tensor,
    cfg: CalibrationConfig,
) -> List:
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    selected: List = []
    selected_names = set()

    # 1) Buffer views: prioritize dropped views from split file.
    dropped = []
    if getattr(dataset, "split_strategy", "") == "file":
        dropped = _load_split_dropped(getattr(dataset, "split_file", ""))
    if len(dropped) > 0:
        by_name = {}
        for cam in test_cams + train_cams:
            by_name[_normalize_name(getattr(cam, "image_name", ""))] = cam
        # Try direct hit first.
        for name in dropped:
            cam = by_name.get(name, None)
            if cam is not None and name not in selected_names:
                selected.append(cam)
                selected_names.add(name)
                if len(selected) >= int(cfg.num_buffer_views):
                    break
        # If still missing, attempt to recover from full camera list.
        if len(selected) < int(cfg.num_buffer_views):
            all_cams = _ensure_all_cameras(scene=scene, dataset=dataset)
            by_name_all = {_normalize_name(getattr(c, "image_name", "")): c for c in all_cams}
            for name in dropped:
                if name in selected_names:
                    continue
                cam = by_name_all.get(name, None)
                if cam is None:
                    continue
                selected.append(cam)
                selected_names.add(name)
                if len(selected) >= int(cfg.num_buffer_views):
                    break

    # Buffer fallback to test views if dropped buffer unavailable.
    if len(selected) < int(cfg.num_buffer_views):
        for cam in test_cams:
            name = _normalize_name(getattr(cam, "image_name", ""))
            if name in selected_names:
                continue
            selected.append(cam)
            selected_names.add(name)
            if len(selected) >= int(cfg.num_buffer_views):
                break

    # 2) Add hardest train views.
    pool_size = min(len(train_cams), int(cfg.hard_view_pool_size))
    pool = train_cams[:pool_size]
    hardness = []
    with torch.no_grad():
        for cam in pool:
            pkg = render_func(cam, triangles, pipe, background)
            h = _compute_view_difficulty(pkg, cam)
            hardness.append(h)
    if len(hardness) > 0:
        order = np.argsort(np.asarray(hardness))[::-1]
        n_hard = int(cfg.num_hard_train_views)
        for idx in order[:n_hard]:
            cam = pool[int(idx)]
            name = _normalize_name(getattr(cam, "image_name", ""))
            if name in selected_names:
                continue
            selected.append(cam)
            selected_names.add(name)

    return selected


def _compute_metrics(render_pkg: Dict, view) -> Dict[str, float]:
    img = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
    gt = torch.clamp(view.original_image.to(img.device), 0.0, 1.0)

    mae = torch.mean(torch.abs(img - gt))
    mse = torch.mean((img - gt) ** 2)
    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-8))

    absrel = torch.tensor(float("nan"), device=img.device)
    delta_125 = torch.tensor(float("nan"), device=img.device)
    if getattr(view, "invdepthmap", None) is not None:
        pred_depth = render_pkg["surf_depth"].detach()[0]
        gt_inv = view.invdepthmap.to(pred_depth.device)[0]
        gt_depth = 1.0 / torch.clamp(gt_inv, min=1e-6)
        valid = torch.isfinite(gt_depth) & torch.isfinite(pred_depth) & (gt_depth > 1e-6) & (pred_depth > 1e-6)
        if torch.any(valid):
            pd = pred_depth[valid]
            gd = gt_depth[valid]
            dm = depth_metrics(
                pred=pd.detach().cpu().numpy().astype(np.float64),
                gt=gd.detach().cpu().numpy().astype(np.float64),
            )
            absrel = torch.tensor(float(dm["abs_rel"]), dtype=torch.float32, device=img.device)
            delta_125 = torch.tensor(float(dm["delta_1.25"]), dtype=torch.float32, device=img.device)

    mean_angle = torch.tensor(float("nan"), device=img.device)
    abs_cos = torch.tensor(float("nan"), device=img.device)
    if getattr(view, "normal_map", None) is not None:
        rn = F.normalize(render_pkg["rend_normal"].detach(), dim=0, eps=1e-6)
        gn = F.normalize(view.normal_map.to(rn.device), dim=0, eps=1e-6)
        c = torch.clamp(torch.abs(torch.sum(rn * gn, dim=0)), 0.0, 1.0)
        abs_cos = torch.mean(c)
        mean_angle = torch.mean(torch.rad2deg(torch.arccos(c)))

    return {
        "psnr": float(psnr.item()),
        "mae": float(mae.item()),
        "absrel": float(absrel.item()) if torch.isfinite(absrel) else float("nan"),
        "delta_125": float(delta_125.item()) if torch.isfinite(delta_125) else float("nan"),
        "mean_normal_angle": float(mean_angle.item()) if torch.isfinite(mean_angle) else float("nan"),
        "abs_cos": float(abs_cos.item()) if torch.isfinite(abs_cos) else float("nan"),
        "image": img,
    }


def _aggregate_metrics(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = ["psnr", "mae", "absrel", "delta_125", "mean_normal_angle", "abs_cos", "changed_pixel_ratio"]
    out = {}
    for k in keys:
        vals = [it[k] for it in items if k in it and np.isfinite(it[k])]
        out[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    return out


def run_counterfactual_simulation(
    scene,
    triangles,
    render_func,
    pipe,
    background: torch.Tensor,
    candidate_triangle_ids: torch.Tensor,
    calibration_views: Sequence,
    gate_cfg: CounterfactualGateConfig,
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
        )
    if len(calibration_views) == 0:
        return CounterfactualDecision(
            accept=False,
            num_candidates=int(cand.numel()),
            deltas={},
            baseline={},
            counterfactual={},
            reason="empty_calibration_set",
        )

    baseline_items = []
    with torch.no_grad():
        for v in calibration_views:
            pkg = render_func(v, triangles, pipe, background)
            m = _compute_metrics(pkg, v)
            baseline_items.append(m)

    cf_items = []
    with TemporaryTriangleMask(triangles=triangles, inactive_triangle_ids=cand):
        with torch.no_grad():
            for i, v in enumerate(calibration_views):
                pkg = render_func(v, triangles, pipe, background)
                m = _compute_metrics(pkg, v)
                base_img = baseline_items[i]["image"]
                cf_img = m["image"]
                diff = torch.mean(torch.abs(base_img - cf_img), dim=0)
                changed = torch.mean((diff > float(gate_cfg.changed_pixel_threshold)).to(torch.float32))
                m["changed_pixel_ratio"] = float(changed.item())
                cf_items.append(m)

    baseline = _aggregate_metrics(baseline_items)
    counterfactual = _aggregate_metrics(cf_items)
    deltas = {
        "delta_psnr": counterfactual["psnr"] - baseline["psnr"] if np.isfinite(counterfactual["psnr"]) and np.isfinite(baseline["psnr"]) else float("nan"),
        "delta_mae": counterfactual["mae"] - baseline["mae"] if np.isfinite(counterfactual["mae"]) and np.isfinite(baseline["mae"]) else float("nan"),
        "delta_absrel": counterfactual["absrel"] - baseline["absrel"] if np.isfinite(counterfactual["absrel"]) and np.isfinite(baseline["absrel"]) else float("nan"),
        "delta_mean_angle": counterfactual["mean_normal_angle"] - baseline["mean_normal_angle"]
        if np.isfinite(counterfactual["mean_normal_angle"]) and np.isfinite(baseline["mean_normal_angle"])
        else float("nan"),
        "changed_pixel_ratio": counterfactual["changed_pixel_ratio"],
    }

    checks = []
    if np.isfinite(deltas["delta_psnr"]):
        checks.append(deltas["delta_psnr"] >= float(gate_cfg.min_delta_psnr_db))
    if np.isfinite(deltas["delta_mae"]):
        checks.append(deltas["delta_mae"] <= float(gate_cfg.max_delta_mae))
    if np.isfinite(deltas["delta_absrel"]):
        checks.append(deltas["delta_absrel"] <= float(gate_cfg.max_delta_absrel))
    if np.isfinite(deltas["delta_mean_angle"]):
        checks.append(deltas["delta_mean_angle"] <= float(gate_cfg.max_delta_mean_angle_deg))
    if np.isfinite(deltas["changed_pixel_ratio"]):
        checks.append(deltas["changed_pixel_ratio"] <= float(gate_cfg.max_changed_pixel_ratio))

    accept = bool(all(checks)) if len(checks) > 0 else False
    return CounterfactualDecision(
        accept=accept,
        num_candidates=int(cand.numel()),
        deltas=deltas,
        baseline=baseline,
        counterfactual=counterfactual,
        reason="ok",
    )


def select_prism_candidate_ids(
    scores: PrismScoreOutputs,
    dead_prune_ratio: float,
    candidate_prune_ratio: float,
) -> torch.Tensor:
    t = int(scores.prune_score_t.numel())
    if t == 0:
        return torch.zeros((0,), dtype=torch.int64, device=scores.prune_score_t.device)

    dead_ids = torch.nonzero(scores.dead_mask, as_tuple=True)[0]
    cand_ids = torch.nonzero(scores.candidate_mask, as_tuple=True)[0]

    dead_n = int(max(0, dead_prune_ratio) * t)
    cand_n = int(max(0, candidate_prune_ratio) * t)

    selected = []
    if dead_ids.numel() > 0 and dead_n > 0:
        s = scores.prune_score_t[dead_ids]
        k = min(dead_n, int(dead_ids.numel()))
        _, idx = torch.topk(s, k=k, largest=True, sorted=False)
        selected.append(dead_ids[idx])
    if cand_ids.numel() > 0 and cand_n > 0:
        s = scores.prune_score_t[cand_ids]
        k = min(cand_n, int(cand_ids.numel()))
        _, idx = torch.topk(s, k=k, largest=True, sorted=False)
        selected.append(cand_ids[idx])

    if len(selected) == 0:
        return torch.zeros((0,), dtype=torch.int64, device=scores.prune_score_t.device)
    return torch.unique(torch.cat(selected, dim=0))
