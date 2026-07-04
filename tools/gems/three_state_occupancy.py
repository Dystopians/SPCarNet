#!/usr/bin/env python
"""GEMS Stage-3 R3-FINAL V1: train-evidence three-state occupancy.

This is the final sanctioned downstream-consumer class in
docs/GEMS_Stage3_Closure_Prompt.md. It builds a static voxel map from a
checkpoint's own train-view median-depth renders:

  FREE      log-odds <= theta_free and evidence >= v_min
  OCCUPIED  log-odds >= theta_occ  and evidence >= v_min
  UNKNOWN   everything else

Planner semantics are deliberately conservative: only FREE voxels are
traversable; OCCUPIED and UNKNOWN are obstacles. Thresholds and the planner
inflation radius are calibrated once on toy_parking clean30k, then frozen and
applied unchanged elsewhere.

The tool does not edit checkpoints, does not use test poses/images, and does
not implement any per-view or per-query selector at consumption time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.gems.downstream_metrics import _VoxelGrid, _build_gt_occupancy  # noqa: E402
from tools.gems.occupancy_routes import build_gt_arg  # noqa: E402
from tools.gems.paired_bootstrap import paired_bootstrap_ci  # noqa: E402
from tools.gems.scenes import SCENES  # noqa: E402

import tools.gems.planner_loop as planner  # noqa: E402


GEMS_ROOT = "/data/peilincai/gems_stage1"
OUT_ROOT_DEFAULT = os.path.join(GEMS_ROOT, "analysis", "r3final_three_state_v1")

VOXEL_M = 0.10
ALPHA_MIN = 0.5
RAY_STRIDE_DEFAULT = 16
FREE_TRUNCATION = 0.95
RAY_STEP_FRACTION = 0.75
W_FREE = -1.0
W_OCC = 2.0

THETA_FREE_GRID = (-0.5, -1.0, -2.0)
THETA_OCC_GRID = (1.0, 2.0)
V_MIN_GRID = (1, 2)
R_INF_GRID = (1.0, 1.2)

CALIB_MAX_PLANNER_CANDIDATES = 4

CHECKPOINTS = {
    ("toy_parking", "clean30k"): os.path.join(
        GEMS_ROOT, "models", "toy_parking_clean30k",
        "point_cloud", "iteration_30000", "point_cloud_state_dict.pt"),
    ("toy_parking", "B50"): os.path.join(
        GEMS_ROOT, "models", "toy_parking_B50_importance_ft_e1v2",
        "point_cloud", "iteration_40000", "point_cloud_state_dict.pt"),
    ("courtyard", "clean30k"): os.path.join(
        GEMS_ROOT, "models", "courtyard_clean30k",
        "point_cloud", "iteration_30000", "point_cloud_state_dict.pt"),
    ("courtyard", "B50"): os.path.join(
        GEMS_ROOT, "models", "courtyard_B50_importance_ft_e1v2",
        "point_cloud", "iteration_40000", "point_cloud_state_dict.pt"),
    ("ss3dm_town01", "clean30k"): os.path.join(
        GEMS_ROOT, "models", "ss3dm_town01_clean30k",
        "point_cloud", "iteration_30000", "point_cloud_state_dict.pt"),
    ("ss3dm_town01", "B50"): os.path.join(
        GEMS_ROOT, "models", "ss3dm_town01_B50_geo_v1",
        "point_cloud", "iteration_40000", "point_cloud_state_dict.pt"),
    ("ss3dm_town02", "clean30k"): os.path.join(
        GEMS_ROOT, "models", "ss3dm_town02_clean30k",
        "point_cloud", "iteration_30000", "point_cloud_state_dict.pt"),
    ("ss3dm_town02", "B50"): os.path.join(
        GEMS_ROOT, "models", "ss3dm_town02_B50_geo_v1",
        "point_cloud", "iteration_40000", "point_cloud_state_dict.pt"),
    ("ss3dm_town03", "clean30k"): os.path.join(
        GEMS_ROOT, "models", "ss3dm_town03_clean30k",
        "point_cloud", "iteration_30000", "point_cloud_state_dict.pt"),
    ("ss3dm_town03", "B50"): os.path.join(
        GEMS_ROOT, "models", "ss3dm_town03_B50_geo_v1",
        "point_cloud", "iteration_40000", "point_cloud_state_dict.pt"),
}


@dataclass(frozen=True)
class Params:
    theta_free: float
    theta_occ: float
    v_min: int
    r_inf: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "theta_free": float(self.theta_free),
            "theta_occ": float(self.theta_occ),
            "v_min": int(self.v_min),
            "r_inf": float(self.r_inf),
        }


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=1)


def _write_md(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _grid_for(scene_name: str) -> _VoxelGrid:
    spec = SCENES[scene_name]
    if spec.roi is None:
        raise ValueError(f"scene {scene_name} has no ROI; cannot build occupancy")
    return _VoxelGrid(spec.roi, VOXEL_M)


def _gt_occ(scene_name: str, grid: _VoxelGrid):
    return _build_gt_occupancy(grid, build_gt_arg(SCENES[scene_name]))


def _indices_from_points(grid: _VoxelGrid, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    idx = np.floor((points - grid.origin[None, :]) / grid.voxel).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < grid.shape[None, :]), axis=1)
    return idx[ok]


def _accumulate_cells(log_odds, evidence, idx, weight):
    if idx.size == 0:
        return
    key = (idx[:, 0].astype(np.int64) * int(log_odds.shape[1]) *
           int(log_odds.shape[2]) +
           idx[:, 1].astype(np.int64) * int(log_odds.shape[2]) +
           idx[:, 2].astype(np.int64))
    uniq = np.unique(key)
    ii = uniq // (int(log_odds.shape[1]) * int(log_odds.shape[2]))
    rem = uniq - ii * int(log_odds.shape[1]) * int(log_odds.shape[2])
    jj = rem // int(log_odds.shape[2])
    kk = rem - jj * int(log_odds.shape[2])
    log_odds[ii, jj, kk] += float(weight)
    evidence[ii, jj, kk] += 1


def pixel_depth_to_world(cam, xs, ys, depth):
    """Back-project pixel centers with camera-z depth to world coordinates.

    Self-test below verifies this against triangle_renderer's projection
    function to <1e-3 px on the repository camera convention.
    """
    W = float(cam.image_width)
    H = float(cam.image_height)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    ndc_x = (2.0 * xs + 1.0) / W - 1.0
    ndc_y = (2.0 * ys + 1.0) / H - 1.0
    x_cam = ndc_x * math.tan(float(cam.FoVx) * 0.5) * depth
    y_cam = ndc_y * math.tan(float(cam.FoVy) * 0.5) * depth
    pts_cam = np.stack([x_cam, y_cam, depth, np.ones_like(depth)], axis=1)
    c2w = np.linalg.inv(cam.world_view_transform.T.detach().cpu().numpy())
    return (pts_cam @ c2w.T)[:, :3]


def _carve_ray(grid, log_odds, evidence, center, hit):
    ray = hit - center
    length = float(np.linalg.norm(ray))
    if not np.isfinite(length) or length <= 1e-6:
        return
    free_len = FREE_TRUNCATION * length
    n_steps = max(1, int(math.ceil(free_len / (grid.voxel * RAY_STEP_FRACTION))))
    ts = np.linspace(0.0, FREE_TRUNCATION, n_steps + 1, endpoint=True)[1:]
    free_pts = center[None, :] + ts[:, None] * ray[None, :]
    _accumulate_cells(log_odds, evidence, _indices_from_points(grid, free_pts), W_FREE)
    _accumulate_cells(log_odds, evidence, _indices_from_points(grid, hit[None, :]), W_OCC)


def build_log_odds_map(checkpoint_path, scene_name, out_dir, ray_stride, max_train_views=None):
    from tools.gems.eval_context import build_eval_context
    import torch

    os.makedirs(out_dir, exist_ok=True)
    spec = SCENES[scene_name]
    grid = _grid_for(scene_name)
    npz_path = os.path.join(out_dir, "log_odds_evidence.npz")
    meta_path = os.path.join(out_dir, "map_meta.json")
    if os.path.exists(npz_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        expected_views = meta.get("n_train_views_used")
        max_views_ok = (
            max_train_views is None or int(expected_views) == int(max_train_views))
        if int(meta.get("ray_stride", -1)) == int(ray_stride) and max_views_ok:
            d = np.load(npz_path)
            print(f"[r3final] reusing cached map {npz_path}", flush=True)
            return grid, d["log_odds"], d["evidence"], meta

    ctx = build_eval_context(checkpoint_path, spec, data_device="cpu")
    log_odds = np.zeros(tuple(grid.shape), dtype=np.float32)
    evidence = np.zeros(tuple(grid.shape), dtype=np.uint16)

    cams = list(ctx.train_cams)
    if max_train_views is not None:
        n = min(int(max_train_views), len(cams))
        keep = np.unique(np.linspace(0, len(cams) - 1, n).round().astype(int))
        cams = [cams[int(i)] for i in keep]
    view_records = []
    t0 = time.time()
    for view_idx, cam in enumerate(cams):
        pkg = ctx.render_view(cam)
        depth = pkg["surf_depth"].detach()[0].cpu().numpy()
        alpha = pkg["rend_alpha"].detach()[0].cpu().numpy()
        H, W = depth.shape
        ys = np.arange(0, H, int(ray_stride), dtype=np.int32)
        xs = np.arange(0, W, int(ray_stride), dtype=np.int32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        d = depth[yy, xx]
        a = alpha[yy, xx]
        valid = np.isfinite(d) & (d > 0.0) & (a >= ALPHA_MIN)
        xs_v = xx[valid].reshape(-1)
        ys_v = yy[valid].reshape(-1)
        d_v = d[valid].reshape(-1)
        hits = pixel_depth_to_world(cam, xs_v, ys_v, d_v)
        center = cam.camera_center.detach().cpu().numpy().astype(np.float64)
        n_before = int(evidence.sum())
        for hit in hits:
            _carve_ray(grid, log_odds, evidence, center, hit)
        touched = int(evidence.sum()) - n_before
        view_records.append({
            "view_index": int(view_idx),
            "image_name": str(getattr(cam, "image_name", view_idx)),
            "sampled_pixels": int(xx.size),
            "valid_rays": int(hits.shape[0]),
            "evidence_votes_added": touched,
        })
        del pkg
        torch.cuda.empty_cache()
        print(f"[r3final] {scene_name} view {view_idx+1}/{len(cams)}: "
              f"{hits.shape[0]} rays, +{touched} evidence votes")

    np.savez_compressed(npz_path, log_odds=log_odds, evidence=evidence,
                        origin=grid.origin, shape=grid.shape, voxel=grid.voxel)
    meta = {
        "scene": scene_name,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "ray_stride": int(ray_stride),
        "alpha_min": ALPHA_MIN,
        "free_truncation": FREE_TRUNCATION,
        "ray_step_fraction": RAY_STEP_FRACTION,
        "w_free": W_FREE,
        "w_occ": W_OCC,
        "n_train_views_used": len(cams),
        "grid_shape": [int(v) for v in grid.shape],
        "wall_s": time.time() - t0,
        "views": view_records,
        "npz": npz_path,
    }
    _write_json(os.path.join(out_dir, "map_meta.json"), meta)
    return grid, log_odds, evidence, meta


def states_from_params(log_odds, evidence, params: Params):
    enough = evidence >= int(params.v_min)
    free = enough & (log_odds <= float(params.theta_free))
    occ = enough & (log_odds >= float(params.theta_occ))
    unknown = ~(free | occ)
    return free, occ, unknown


def free_set_confusion(free, gt_occ):
    gt = gt_occ.astype(bool)
    ff = free[gt]
    blocked = (~free)[~gt]
    out = {
        "n_voxels": int(free.size),
        "n_gt_occ": int(gt.sum()),
        "n_gt_free": int((~gt).sum()),
        "free_at_gt_occ_rate": float(ff.mean()) if ff.size else None,
        "blocked_at_gt_free_rate": float(blocked.mean()) if blocked.size else None,
        "free_fraction": float(free.mean()),
    }
    return out


def state_fractions(free, occ, unknown):
    n = float(free.size)
    return {
        "free_fraction": float(free.sum() / n),
        "occupied_fraction": float(occ.sum() / n),
        "unknown_fraction": float(unknown.sum() / n),
    }


def _set_planner_inflation(r_inf):
    r = float(r_inf)
    if r <= planner.HALF_WIDTH_M:
        raise ValueError(f"r_inf={r} must exceed vehicle half width {planner.HALF_WIDTH_M}")
    planner.INFLATE_R_M = r
    planner.SPINE_HALF_M = (
        planner.VEHICLE_LENGTH_M / 2.0 -
        math.sqrt(r ** 2 - planner.HALF_WIDTH_M ** 2)
    )


def _planner_setup(scene_name, gt_occ, r_inf):
    _set_planner_inflation(r_inf)
    grid = _grid_for(scene_name)
    ptab = planner.PrimitiveTable()
    gt_band = planner.footprint_layer(gt_occ, grid)
    maps_gt = planner.GridMaps(gt_band, ptab)
    rng = np.random.default_rng(planner.SEED)
    problems = planner.sample_problems(maps_gt, rng)
    goal_cells = [p["goal"][0] * maps_gt.ny + p["goal"][1] for p in problems]
    dist_ref = maps_gt.dijkstra_from(goal_cells)
    metrics_ref, per_ref = planner.plan_cell(
        f"{scene_name}__GTREF__r{r_inf}", maps_gt, gt_band, problems, ptab, dist_ref)
    return grid, ptab, gt_band, maps_gt, problems, metrics_ref, per_ref


def _plan_free_map(scene_name, free, gt_occ, params: Params, out_dir, label,
                   setup=None):
    if setup is None:
        setup = _planner_setup(scene_name, gt_occ, params.r_inf)
    grid, ptab, gt_band, maps_gt, problems, metrics_ref, per_ref = setup
    blocked3d = ~free
    band = planner.footprint_layer(blocked3d, grid)
    maps = planner.GridMaps(band, ptab)
    goal_cells = [p["goal"][0] * maps.ny + p["goal"][1] for p in problems]
    dist_maps = maps.dijkstra_from(goal_cells)
    metrics, per_problem = planner.plan_cell(
        f"{scene_name}__{label}", maps, gt_band, problems, ptab, dist_maps)
    metrics["path_length_inflation_vs_gtref"] = planner.path_length_inflation(
        per_problem, per_ref)
    metrics["gtref"] = {
        "plans_found": metrics_ref["plans_found"],
        "collisions_per_100_plans": metrics_ref["collisions_per_100_plans"],
        "mean_path_length_m": metrics_ref["mean_path_length_m"],
    }
    clean_pp = [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in per_problem]
    _write_json(os.path.join(out_dir, "cell_metrics.json"),
                {"metrics": metrics, "per_problem": clean_pp})
    return metrics, per_problem, metrics_ref, per_ref


def _toy_bar(metrics, gtref_metrics):
    found = int(metrics["plans_found"])
    gt_found = int(gtref_metrics["plans_found"])
    coll = metrics["collisions_per_100_plans"]
    infl = metrics.get("path_length_inflation_vs_gtref", {})
    med_infl = infl.get("median_inflation")
    return (
        found >= math.ceil(0.5 * gt_found) and
        coll is not None and float(coll) <= 3.0 and
        med_infl is not None and float(med_infl) <= 1.5
    )


def calibrate_toy_clean(out_root, ray_stride, max_train_views=None):
    scene_name, model_label = "toy_parking", "clean30k"
    calibration_path = os.path.join(out_root, "calibration", "toy_clean_calibration.json")
    if os.path.exists(calibration_path):
        with open(calibration_path) as f:
            report = json.load(f)
        params = Params(**report["selected_params"])
        print(f"[r3final] reusing cached calibration {calibration_path}", flush=True)
        return params, report
    cdir = os.path.join(out_root, "maps", f"{scene_name}__{model_label}")
    ckpt = CHECKPOINTS[(scene_name, model_label)]
    grid, log_odds, evidence, meta = build_log_odds_map(
        ckpt, scene_name, cdir, ray_stride, max_train_views=max_train_views)
    gt_occ = _gt_occ(scene_name, grid)

    all_candidates = []
    for tf in THETA_FREE_GRID:
        for to in THETA_OCC_GRID:
            for vm in V_MIN_GRID:
                for ri in R_INF_GRID:
                    p = Params(tf, to, vm, ri)
                    free, occ, unk = states_from_params(log_odds, evidence, p)
                    conf = free_set_confusion(free, gt_occ)
                    frac = state_fractions(free, occ, unk)
                    p1_ok = (
                        conf["free_at_gt_occ_rate"] is not None and
                        conf["free_at_gt_occ_rate"] <= 0.10
                    )
                    all_candidates.append({
                        "params": p.as_dict(),
                        "p1_free_false_free_ok": bool(p1_ok),
                        "confusion": conf,
                        "state_fractions": frac,
                    })

    eligible = [c for c in all_candidates if c["p1_free_false_free_ok"]]
    pool = eligible if eligible else all_candidates
    pool = sorted(pool, key=lambda c: (
        0 if c["p1_free_false_free_ok"] else 1,
        c["confusion"]["blocked_at_gt_free_rate"]
        if c["confusion"]["blocked_at_gt_free_rate"] is not None else 1.0,
        c["state_fractions"]["unknown_fraction"],
        -c["state_fractions"]["free_fraction"],
        abs(c["params"]["theta_free"]),
        c["params"]["v_min"],
        c["params"]["r_inf"],
    ))
    planner_candidates = pool[:CALIB_MAX_PLANNER_CANDIDATES]

    planned = []
    gtref_by_r = {}
    setup_cache = {}
    for i, cand in enumerate(planner_candidates):
        p = Params(**cand["params"])
        free, occ, unk = states_from_params(log_odds, evidence, p)
        pdir = os.path.join(out_root, "calibration", f"candidate_{i:02d}")
        setup = setup_cache.get(p.r_inf)
        if setup is None:
            setup = _planner_setup(scene_name, gt_occ, p.r_inf)
            setup_cache[p.r_inf] = setup
        metrics, per_problem, gtref_metrics, per_ref = _plan_free_map(
            scene_name, free, gt_occ, p, pdir, "toy_calib", setup=setup)
        cand = dict(cand)
        cand["planner_metrics"] = metrics
        cand["toy_bar_pass"] = _toy_bar(metrics, gtref_metrics)
        planned.append(cand)
        gtref_by_r[str(p.r_inf)] = gtref_metrics
        print(f"[r3final] calib cand {i}: params={p.as_dict()} "
              f"found={metrics['plans_found']}/100 coll={metrics['collisions_per_100_plans']} "
              f"toy_bar={cand['toy_bar_pass']}")

    def rank_planned(c):
        m = c["planner_metrics"]
        infl = m.get("path_length_inflation_vs_gtref", {})
        med = infl.get("median_inflation")
        coll = m["collisions_per_100_plans"]
        return (
            0 if c["toy_bar_pass"] else 1,
            -int(m["plans_found"]),
            float(coll) if coll is not None else 999.0,
            float(med) if med is not None else 999.0,
            c["confusion"]["free_at_gt_occ_rate"]
            if c["confusion"]["free_at_gt_occ_rate"] is not None else 999.0,
        )

    selected = sorted(planned, key=rank_planned)[0]
    params = Params(**selected["params"])
    report = {
        "pre_registration": {
            "mechanism": "V1 log-odds visibility carving",
            "calibration_scene": "toy_parking clean30k only",
            "ray_stride": int(ray_stride),
            "alpha_min": ALPHA_MIN,
            "free_truncation": FREE_TRUNCATION,
            "weights": {"free": W_FREE, "occupied": W_OCC},
            "threshold_grid": {
                "theta_free": list(THETA_FREE_GRID),
                "theta_occ": list(THETA_OCC_GRID),
                "v_min": list(V_MIN_GRID),
                "r_inf": list(R_INF_GRID),
            },
            "planner_candidate_cap": CALIB_MAX_PLANNER_CANDIDATES,
            "planner_candidate_policy": (
                "filter by toy FREE false-free <=10%; sort by lower "
                "blocked_at_gt_free, lower unknown fraction, higher free "
                "fraction; planner-evaluate the first K; select a toy-bar "
                "PASS if any, else the highest found/lower collision candidate"),
        },
        "all_confusion_candidates": all_candidates,
        "planner_candidates": planned,
        "selected": selected,
        "selected_params": params.as_dict(),
        "gtref_by_r": gtref_by_r,
        "map_meta": meta,
    }
    _write_json(os.path.join(out_root, "calibration", "toy_clean_calibration.json"), report)
    return params, report


def _paired_found_ci(per_a, per_b):
    a = np.array([1.0 if r["found"] else 0.0 for r in per_a], dtype=np.float64)
    b = np.array([1.0 if r["found"] else 0.0 for r in per_b], dtype=np.float64)
    ci = paired_bootstrap_ci(a, b)
    ci["rate_a"] = float(a.mean())
    ci["rate_b"] = float(b.mean())
    return ci


def _paired_collision_ci(per_a, per_b):
    a, b = [], []
    for ra, rb in zip(per_a, per_b):
        if ra["found"] and rb["found"]:
            a.append(1.0 if ra["gt_collision"] else 0.0)
            b.append(1.0 if rb["gt_collision"] else 0.0)
    if len(a) < 2:
        return {"n_common_found": len(a), "unavailable": True}
    ci = paired_bootstrap_ci(np.asarray(a), np.asarray(b))
    ci["n_common_found"] = len(a)
    ci["rate_a"] = float(np.mean(a))
    ci["rate_b"] = float(np.mean(b))
    return ci


def evaluate_cell(scene_name, model_label, params: Params, out_root, ray_stride,
                  max_train_views=None, setup_cache=None):
    cell_json = os.path.join(out_root, "cells", f"{scene_name}__{model_label}.json")
    planner_json = os.path.join(
        out_root, "planner", f"{scene_name}__{model_label}", "cell_metrics.json")
    if os.path.exists(cell_json) and os.path.exists(planner_json):
        with open(cell_json) as f:
            record = json.load(f)
        if record.get("params") == params.as_dict():
            with open(planner_json) as f:
                pp = json.load(f).get("per_problem", [])
            print(f"[r3final] reusing cached cell {scene_name}/{model_label}", flush=True)
            return record, pp
    ckpt = CHECKPOINTS[(scene_name, model_label)]
    cdir = os.path.join(out_root, "maps", f"{scene_name}__{model_label}")
    grid, log_odds, evidence, meta = build_log_odds_map(
        ckpt, scene_name, cdir, ray_stride, max_train_views=max_train_views)
    gt_occ = _gt_occ(scene_name, grid)
    free, occ, unk = states_from_params(log_odds, evidence, params)
    conf = free_set_confusion(free, gt_occ)
    frac = state_fractions(free, occ, unk)
    np.savez_compressed(os.path.join(cdir, "states_selected.npz"),
                        free=free, occupied=occ, unknown=unk,
                        log_odds=log_odds, evidence=evidence)
    setup = None
    if setup_cache is not None:
        key = (scene_name, float(params.r_inf))
        setup = setup_cache.get(key)
        if setup is None:
            setup = _planner_setup(scene_name, gt_occ, params.r_inf)
            setup_cache[key] = setup
    metrics, per_problem, gtref_metrics, per_ref = _plan_free_map(
        scene_name, free, gt_occ, params,
        os.path.join(out_root, "planner", f"{scene_name}__{model_label}"),
        model_label, setup=setup)
    record = {
        "scene": scene_name,
        "model_label": model_label,
        "checkpoint_path": os.path.abspath(ckpt),
        "params": params.as_dict(),
        "map_meta": meta,
        "confusion": conf,
        "state_fractions": frac,
        "planner_metrics": metrics,
        "toy_bar_pass": (
            _toy_bar(metrics, gtref_metrics) if scene_name == "toy_parking" else None),
        "courtyard_fix_target_pass": (
            (metrics["plans_found"] >= 30 and
             metrics["collisions_per_100_plans"] is not None and
             float(metrics["collisions_per_100_plans"]) <= 3.0)
            if scene_name == "courtyard" else None),
    }
    _write_json(os.path.join(out_root, "cells", f"{scene_name}__{model_label}.json"), record)
    return record, per_problem


def write_summary(out_root, calibration, cells, comparisons):
    courtyard = [
        c for c in cells
        if c["scene"] == "courtyard" and c["model_label"] in ("clean30k", "B50")
    ]
    toy = [
        c for c in cells
        if c["scene"] == "toy_parking" and c["model_label"] in ("clean30k", "B50")
    ]
    courtyard_pass = bool(courtyard) and all(
        c.get("courtyard_fix_target_pass") for c in courtyard)
    toy_pass = bool(toy) and all(c.get("toy_bar_pass") for c in toy)
    verdict = {
        "mechanism": "V1 log-odds visibility carving",
        "stage3_r3_final_v1_pass": bool(courtyard_pass and toy_pass),
        "toy_bar_pass_all_reported": bool(toy_pass),
        "courtyard_fix_target_pass_all_reported": bool(courtyard_pass),
        "selected_params": calibration["selected_params"],
        "note": (
            "PASS requires toy found >=0.5x GTREF at <=3 collisions/100 and "
            "median path inflation <=1.5x, plus courtyard >=30/100 found and "
            "<=3 collisions/100. If false, Stage3 may consume V2/V3 only if "
            "pre-registered as near-miss mechanisms; otherwise write the "
            "impossibility addendum after <=3 mechanisms total."),
    }
    summary = {
        "verdict": verdict,
        "calibration": calibration,
        "cells": cells,
        "comparisons": comparisons,
    }
    _write_json(os.path.join(out_root, "summary.json"), summary)

    lines = [
        "# R3-FINAL V1 — Three-State Evidence-Carved Occupancy",
        "",
        f"Selected params: `{json.dumps(calibration['selected_params'], sort_keys=True)}`.",
        "",
        "| Scene | Model | FREE frac | OCC frac | UNKNOWN frac | FREE@GT-occ | blocked@GT-free | Found/100 | Coll/100 | Median infl. | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in cells:
        pm = c["planner_metrics"]
        infl = pm.get("path_length_inflation_vs_gtref", {})
        verdict_cell = "PASS" if (c.get("toy_bar_pass") or c.get("courtyard_fix_target_pass")) else "FAIL/diag"
        lines.append(
            f"| {c['scene']} | {c['model_label']} | "
            f"{c['state_fractions']['free_fraction']:.4f} | "
            f"{c['state_fractions']['occupied_fraction']:.4f} | "
            f"{c['state_fractions']['unknown_fraction']:.4f} | "
            f"{c['confusion']['free_at_gt_occ_rate']:.4f} | "
            f"{c['confusion']['blocked_at_gt_free_rate']:.4f} | "
            f"{pm['plans_found']} | {pm['collisions_per_100_plans']} | "
            f"{infl.get('median_inflation')} | {verdict_cell} |")
    lines += [
        "",
        "## Overall V1 Verdict",
        "",
        f"`stage3_r3_final_v1_pass = {verdict['stage3_r3_final_v1_pass']}`",
        "",
        "## Clean vs B50 Invariance Checks",
        "",
        "These are paired on the same GT-valid 100 seed-0 problems under the frozen consumer.",
        "",
        "```json",
        json.dumps(_json_safe(comparisons), indent=1, sort_keys=True),
        "```",
        "",
    ]
    _write_md(os.path.join(out_root, "R3_FINAL_V1_REPORT.md"), "\n".join(lines))
    return summary


def run_v1(args):
    os.makedirs(args.out_root, exist_ok=True)
    params, calibration = calibrate_toy_clean(
        args.out_root, args.ray_stride, max_train_views=args.max_train_views)
    cells = []
    per_problem = {}
    setup_cache = {}
    for scene_name, model_label in [
        ("toy_parking", "clean30k"),
        ("toy_parking", "B50"),
        ("courtyard", "clean30k"),
        ("courtyard", "B50"),
    ]:
        record, pp = evaluate_cell(
            scene_name, model_label, params, args.out_root, args.ray_stride,
            max_train_views=args.max_train_views, setup_cache=setup_cache)
        cells.append(record)
        per_problem[(scene_name, model_label)] = pp

    comparisons = {}
    for scene_name in ("toy_parking", "courtyard"):
        a = per_problem[(scene_name, "B50")]
        b = per_problem[(scene_name, "clean30k")]
        comparisons[scene_name] = {
            "B50_minus_clean_found": _paired_found_ci(a, b),
            "B50_minus_clean_collision": _paired_collision_ci(a, b),
        }
    return write_summary(args.out_root, calibration, cells, comparisons)


def selftest():
    """CUDA-light projection convention test plus state-threshold semantics."""
    import torch
    from tools.gems.eval_context import _camera_loader_args, _read_scene_info
    from triangle_renderer import compute_image_2d_pytorch_exact
    from utils.camera_utils import cameraList_from_camInfos

    spec = SCENES["toy_parking"]
    info = _read_scene_info(spec)
    cam_args = _camera_loader_args(spec, "cpu")
    cam = cameraList_from_camInfos(info.train_cameras[:1], 1.0, cam_args)[0]
    W, H = int(cam.image_width), int(cam.image_height)
    px = np.array([0.25 * W, 0.5 * W, 0.75 * W], dtype=np.float64)
    py = np.array([0.25 * H, 0.5 * H, 0.75 * H], dtype=np.float64)
    depth = np.array([5.0, 10.0, 15.0], dtype=np.float64)
    pts = pixel_depth_to_world(cam, px, py, depth)
    pix = compute_image_2d_pytorch_exact(
        torch.from_numpy(pts).float().cuda(),
        cam.full_proj_transform.cuda(), W, H).detach().cpu().numpy()
    err = float(np.max(np.abs(pix - np.stack([px, py], axis=1))))
    if err > 1e-3:
        raise AssertionError(f"backprojection convention error too high: {err}")

    log_odds = np.array([-2.0, 0.0, 3.0], dtype=np.float32).reshape(3, 1, 1)
    evidence = np.array([2, 0, 2], dtype=np.uint16).reshape(3, 1, 1)
    free, occ, unk = states_from_params(log_odds, evidence, Params(-1.0, 1.0, 1, 1.0))
    if not (free[0, 0, 0] and unk[1, 0, 0] and occ[2, 0, 0]):
        raise AssertionError("three-state threshold semantics broken")
    print("three_state_occupancy selftest PASSED")


def main():
    ap = argparse.ArgumentParser(description="GEMS Stage3 R3-FINAL V1")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--ray-stride", type=int, default=RAY_STRIDE_DEFAULT)
    ap.add_argument("--max-train-views", type=int, default=None,
                    help="debug only; omit for the registered full train-view run")
    ap.add_argument("--gpu", default=None,
                    help="sets CUDA_VISIBLE_DEVICES before torch context build")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.selftest:
        selftest()
        return
    run_v1(args)


if __name__ == "__main__":
    main()
