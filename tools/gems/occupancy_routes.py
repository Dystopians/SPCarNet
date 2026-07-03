#!/usr/bin/env python
"""GEMS Stage-1R R3.a — occupancy-extraction routes study (LEDGER GOAL #R-02).

Compares two ways to consume a mesh-splat checkpoint as an occupancy grid:

  route (i)  — triangle voxelization: the EXISTING d1 machinery
               (tools/gems/downstream_metrics). Wrapped, not duplicated:
               the same _VoxelGrid / _rasterize_triangles / _build_gt_occupancy
               / _sample_trajectories / _collision_verdicts functions are used.
  route (ii) — TSDF fusion of TRAIN-VIEW rendered median depths (`surf_depth`,
               training-time settings via tools/gems/eval_context): projective
               TSDF sampled at voxel centers, per-voxel robust aggregation =
               median of clamped SDF samples; occupied iff
               n_obs >= min_views AND |median| < iso.

Framing guard (Stage-1R insert §R3): route (ii) produces a ONE-TIME, GLOBAL,
TRAIN-EVIDENCE-ONLY artifact (D4). Nothing varies per view or per query. The
fusion parameters {tau_t, iso, min_views} are calibrated ONCE on toy clean30k
(the sanctioned <=3x3x2 grid, table logged), FROZEN, and applied unchanged to
every other model/scene.

Pre-registration (LEDGER GOAL #R-02, frozen before any route-(ii) number was
computed): route (ii) reduces toy d1 false_free_rate by >=50% relative vs
route (i) at <=2x route-(i) false_occupied_rate; PASS iff all 3 toy models meet
the bar with paired-bootstrap CI excl. 0 on the false-free difference (units =
GT-occupied voxels). Courtyard = frozen-params transfer verdict.

Usage:
    python tools/gems/occupancy_routes.py --gpu 4          # full study
    python tools/gems/occupancy_routes.py --selftest       # numpy-only tests

Outputs (durable): /data/peilincai/gems_stage1/analysis/r3a_occupancy_routes/
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# numpy-only module — safe to import before CUDA_VISIBLE_DEVICES is set.
from tools.gems.downstream_metrics import (          # noqa: E402
    _VoxelGrid,
    _build_gt_occupancy,
    _collision_verdicts,
    _rasterize_triangles,
    _sample_trajectories,
)
from tools.gems.paired_bootstrap import paired_bootstrap_ci  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen study constants (pre-registered in LEDGER GOAL #R-02)
# ---------------------------------------------------------------------------
VOXEL_M = 0.10          # PROTOCOL 4.4 d1 voxel size
N_TRAJ = 200            # PROTOCOL 4.4 d2 trajectory count
SEED = 0                # PROTOCOL 4.4 d2 sampler seed — UNCHANGED
ALPHA_MIN = 0.5         # same gate as g1/g2 (geometry_metrics.G1_ALPHA_MIN)

# Sanctioned one-time calibration grid (<= 3x3x2) — toy clean30k ONLY.
TAU_T_GRID = (0.20, 0.30, 0.45)     # truncation (m)
ISO_GRID = (0.05, 0.10, 0.15)       # |tsdf| < iso occupancy band (m)
MIN_VIEWS_GRID = (2, 5)             # min observations per voxel
TAU_MAX = max(TAU_T_GRID)           # raw samples kept down to -TAU_MAX

# Pre-registered bar.
REL_FF_REDUCTION_BAR = 0.50
FO_MULTIPLE_BAR = 2.0

OUT_ROOT_DEFAULT = "/data/peilincai/gems_stage1/analysis/r3a_occupancy_routes"
MODELS_ROOT = "/data/peilincai/gems_stage1/models"
EVAL_ROOT = "/data/peilincai/gems_stage1/eval"

# (scene, label, checkpoint, existing route-(i) eval row for cross-check)
MODELS = [
    ("toy_parking", "clean30k",
     f"{MODELS_ROOT}/toy_parking_clean30k/point_cloud/iteration_30000/point_cloud_state_dict.pt",
     f"{EVAL_ROOT}/toy_parking_clean30k_v1/metrics.json"),
    ("toy_parking", "B50_importance_ft_e1v2_40000",
     f"{MODELS_ROOT}/toy_parking_B50_importance_ft_e1v2/point_cloud/iteration_40000/point_cloud_state_dict.pt",
     f"{EVAL_ROOT}/toy_parking_B50_importance_ft_e1v2/metrics.json"),
    ("toy_parking", "B25_importance_ft_e1v2_40000",
     f"{MODELS_ROOT}/toy_parking_B25_importance_ft_e1v2/point_cloud/iteration_40000/point_cloud_state_dict.pt",
     f"{EVAL_ROOT}/toy_parking_B25_importance_ft_e1v2/metrics.json"),
    ("courtyard", "clean30k",
     f"{MODELS_ROOT}/courtyard_clean30k/point_cloud/iteration_30000/point_cloud_state_dict.pt",
     f"{EVAL_ROOT}/courtyard_clean30k_v4/metrics.json"),
    ("courtyard", "B50_importance_ft_e1v2_40000",
     f"{MODELS_ROOT}/courtyard_B50_importance_ft_e1v2/point_cloud/iteration_40000/point_cloud_state_dict.pt",
     f"{EVAL_ROOT}/courtyard_B50_importance_ft_e1v2/metrics.json"),
]
CALIBRATION_MODEL = ("toy_parking", "clean30k")   # sanctioned one-time


# ---------------------------------------------------------------------------
# route (i) + GT + d2 — thin wrappers over downstream_metrics (no fork)
# ---------------------------------------------------------------------------

def build_grid(roi):
    return _VoxelGrid(roi, VOXEL_M)


def route_i_occupancy(grid, verts_np, faces_np):
    """PROTOCOL 4.4 d1 recon grid: triangle rasterization, unchanged."""
    occ = grid.new_occupancy()
    _rasterize_triangles(grid, occ, verts_np, faces_np)
    return occ


def gt_occupancy(grid, gt_arg):
    return _build_gt_occupancy(grid, gt_arg)


def sample_trajectories(roi):
    """Identical 200-trajectory set, sampler seed 0 unchanged."""
    return _sample_trajectories(np.random.default_rng(SEED), roi, N_TRAJ)


def collision_verdicts(trajs, grid, occ):
    return _collision_verdicts(trajs, grid, occ)


def d1_confusion(gt_occ, recon_occ):
    """Exact d1 formulas (downstream_metrics.compute_downstream_metrics),
    plus the per-voxel indicator arrays used for pairing."""
    gt_flat = gt_occ.ravel()
    rec_flat = recon_occ.ravel()
    free_at_gt_occ = ~rec_flat[gt_flat]
    occ_at_gt_free = rec_flat[~gt_flat]
    n_gt_occ = int(gt_flat.sum())
    n_gt_free = int(gt_flat.size - n_gt_occ)
    return {
        "false_free_rate": float(free_at_gt_occ.mean()) if n_gt_occ else float("nan"),
        "false_occupied_rate": float(occ_at_gt_free.mean()) if n_gt_free else float("nan"),
        "n_gt_occupied": n_gt_occ,
        "n_gt_free": n_gt_free,
        "n_recon_occupied": int(rec_flat.sum()),
        "free_at_gt_occ": free_at_gt_occ,      # bool [n_gt_occ], voxel-paired
        "occ_at_gt_free": occ_at_gt_free,      # bool [n_gt_free], voxel-paired
    }


# ---------------------------------------------------------------------------
# route (ii) — TSDF fusion of train-view rendered median depths
# ---------------------------------------------------------------------------

def collect_tsdf_samples(ctx, grid, tau_max=TAU_MAX, alpha_min=ALPHA_MIN):
    """One pass over ALL train views (D4): render median depth, sample the
    projective SDF at every voxel center that projects into the view.

    A sample is a valid observation iff: voxel in front of the camera, pixel
    in frame, rendered alpha >= alpha_min, rendered depth finite and > 0, and
    sdf = rendered_depth - voxel_cam_depth > -tau_max (occlusion discard).

    Returns dict: vox_ids int32 [S] (flat voxel index), sdf float32 [S] (RAW,
    unclamped — clamping happens per tau_t at aggregation), n_views, n_voxels,
    per_view_n_samples.
    """
    import torch
    from tools.gems.geometry_metrics import _project_points

    shape = tuple(int(s) for s in grid.shape)
    n_vox = int(np.prod(shape))
    idx = np.arange(n_vox, dtype=np.int64)
    ii, jj, kk = np.unravel_index(idx, shape)
    centers = (grid.origin[None, :]
               + (np.stack([ii, jj, kk], axis=1).astype(np.float64) + 0.5) * grid.voxel)
    centers_t = torch.from_numpy(centers.astype(np.float32)).cuda()
    vox_ids_all = torch.arange(n_vox, dtype=torch.int64, device="cuda")
    del idx, ii, jj, kk, centers

    vox_chunks, sdf_chunks, per_view = [], [], []
    n_views = 0
    with torch.no_grad():
        for cam in ctx.train_cams:
            depth_v, px, py, in_front = _project_points(centers_t, cam)
            W, H = int(cam.image_width), int(cam.image_height)
            ok = in_front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            n_views += 1
            if not bool(ok.any().item()):
                per_view.append(0)
                continue
            pkg = ctx.render_view(cam)
            depth_map = pkg["surf_depth"].detach()[0]   # [H, W] median depth
            alpha_map = pkg["rend_alpha"].detach()[0]   # [H, W]
            px_k, py_k = px[ok], py[ok]
            d_r = depth_map[py_k, px_k]
            a_r = alpha_map[py_k, px_k]
            sdf = d_r - depth_v[ok]
            keep = ((a_r >= alpha_min) & torch.isfinite(d_r) & (d_r > 0.0)
                    & (sdf > -tau_max))
            vox_chunks.append(vox_ids_all[ok][keep].cpu().numpy().astype(np.int32))
            sdf_chunks.append(sdf[keep].cpu().numpy().astype(np.float32))
            per_view.append(int(vox_chunks[-1].shape[0]))
            del pkg, depth_map, alpha_map, d_r, a_r, sdf, keep, px_k, py_k
            del depth_v, px, py, in_front, ok
            torch.cuda.empty_cache()

    vox_ids = (np.concatenate(vox_chunks) if vox_chunks
               else np.empty(0, dtype=np.int32))
    sdf = (np.concatenate(sdf_chunks) if sdf_chunks
           else np.empty(0, dtype=np.float32))
    return {"vox_ids": vox_ids, "sdf": sdf, "n_views": n_views,
            "n_voxels": n_vox, "per_view_n_samples": per_view}


def prepare_sorted_samples(samples):
    """Sort samples by (voxel id, sdf) so each voxel's observations are a
    contiguous ascending-sdf group; all 18 calibration combos reuse this."""
    vox_ids, sdf = samples["vox_ids"], samples["sdf"]
    order = np.lexsort((sdf, vox_ids))
    vox_s = vox_ids[order]
    sdf_s = sdf[order]
    uniq, starts, counts = np.unique(vox_s, return_index=True, return_counts=True)
    return {"uniq": uniq.astype(np.int64), "starts": starts, "counts": counts,
            "sdf_sorted": sdf_s, "n_voxels": samples["n_voxels"],
            "n_samples": int(sdf_s.shape[0]), "n_views": samples["n_views"]}


def tsdf_occupancy(prepared, grid_shape, tau_t, iso, min_views):
    """Fused occupancy for one (tau_t, iso, min_views) combo.

    Per voxel: observations = samples with sdf > -tau_t (within-group ascending
    order makes the invalid ones a prefix); tsdf = median of clamped samples
    (clamp is monotone, so the median of clamped values is read off the two
    middle order statistics of the valid suffix); occupied iff
    n_obs >= min_views AND |tsdf| < iso. Unobserved voxels are free.
    """
    sdf_s = prepared["sdf_sorted"]
    starts, counts, uniq = prepared["starts"], prepared["counts"], prepared["uniq"]
    n_vox = int(np.prod(grid_shape))

    occ_flat = np.zeros(n_vox, dtype=bool)
    if sdf_s.shape[0] == 0:
        return occ_flat.reshape(grid_shape)

    valid = sdf_s > -float(tau_t)
    n_valid = np.add.reduceat(valid.astype(np.int64), starts)
    n_invalid = counts - n_valid
    sel = n_valid >= max(int(min_views), 1)
    if not sel.any():
        return occ_flat.reshape(grid_shape)

    base = starts[sel] + n_invalid[sel]
    nv = n_valid[sel]
    lo = np.clip(sdf_s[base + (nv - 1) // 2], -tau_t, tau_t)
    hi = np.clip(sdf_s[base + nv // 2], -tau_t, tau_t)
    med = 0.5 * (lo.astype(np.float64) + hi.astype(np.float64))
    occ_flat[uniq[sel][np.abs(med) < float(iso)]] = True
    return occ_flat.reshape(grid_shape)


# ---------------------------------------------------------------------------
# GT adapter (replicates run_eval.py's SceneSpec.gt -> downstream contract)
# ---------------------------------------------------------------------------

def build_gt_arg(spec):
    if spec.gt.get("mesh_path") and os.path.isfile(spec.gt["mesh_path"]):
        return {"mesh_path": spec.gt["mesh_path"]}
    import trimesh
    transforms = spec.gt.get("scan_transforms")
    clouds = []
    for i, p in enumerate(spec.gt["scan_paths"]):
        v = np.asarray(trimesh.load(p, process=False).vertices, dtype=np.float64)
        if transforms is not None:
            M = np.asarray(transforms[i], dtype=np.float64)
            v = v @ M[:3, :3].T + M[:3, 3]
        clouds.append(v)
    return {"scan_points": np.concatenate(clouds, axis=0)}


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items() if not isinstance(v, np.ndarray)}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _scene_assets(scene_name):
    from tools.gems.scenes import SCENES
    spec = SCENES[scene_name]
    grid = build_grid(spec.roi)
    gt_occ = gt_occupancy(grid, build_gt_arg(spec))
    trajs = sample_trajectories(spec.roi)
    gt_verdicts = collision_verdicts(trajs, grid, gt_occ)
    return spec, grid, gt_occ, trajs, gt_verdicts


def _model_routes(ctx, grid, gt_occ, trajs, gt_verdicts, frozen, samples=None):
    """Both routes for one model. Returns (record, arrays, samples)."""
    import torch
    finite = ctx.finite_faces_mask()
    verts_np = ctx.vertices().cpu().numpy()
    faces_np = ctx.faces()[finite].cpu().numpy()
    n_nonfinite = int((~finite).sum().item())

    t0 = time.time()
    occ_i = route_i_occupancy(grid, verts_np, faces_np)
    t_route_i = time.time() - t0

    t0 = time.time()
    if samples is None:
        samples = collect_tsdf_samples(ctx, grid)
    prepared = prepare_sorted_samples(samples)
    occ_ii = tsdf_occupancy(prepared, tuple(int(s) for s in grid.shape),
                            frozen["tau_t"], frozen["iso"], frozen["min_views"])
    t_route_ii = time.time() - t0

    conf_i = d1_confusion(gt_occ, occ_i)
    conf_ii = d1_confusion(gt_occ, occ_ii)

    # Paired bootstrap on the false-free difference, units = GT-occupied
    # voxels (pre-registered). mean_diff = ff_i - ff_ii; ci_lo > 0 => route ii
    # significantly reduces false-free.
    ff_ci = paired_bootstrap_ci(conf_i["free_at_gt_occ"].astype(np.float64),
                                conf_ii["free_at_gt_occ"].astype(np.float64))

    v_i = collision_verdicts(trajs, grid, occ_i)
    v_ii = collision_verdicts(trajs, grid, occ_ii)

    def d2_stats(v_recon):
        agree = v_recon == gt_verdicts
        n_gt_coll = int(gt_verdicts.sum())
        unsafe = float((~v_recon[gt_verdicts]).mean()) if n_gt_coll else float("nan")
        return {"agreement_rate": float(agree.mean()),
                "unsafe_disagreement_rate": unsafe,
                "n_recon_collision": int(v_recon.sum()),
                "n_gt_collision": n_gt_coll, "n_traj": len(gt_verdicts),
                "seed": SEED}

    ff_i, ff_ii = conf_i["false_free_rate"], conf_ii["false_free_rate"]
    fo_i, fo_ii = conf_i["false_occupied_rate"], conf_ii["false_occupied_rate"]
    rel_red = (ff_i - ff_ii) / ff_i if ff_i > 0 else float("nan")
    fo_mult = fo_ii / fo_i if fo_i > 0 else float("inf")
    record = {
        "n_triangles_used": int(faces_np.shape[0]),
        "n_nonfinite_faces_excluded": n_nonfinite,
        "route_i": {"d1": _json_safe(conf_i), "d2": d2_stats(v_i),
                    "wallclock_sec": t_route_i},
        "route_ii": {"d1": _json_safe(conf_ii), "d2": d2_stats(v_ii),
                     "frozen_params": frozen,
                     "n_train_views_fused": prepared["n_views"],
                     "n_tsdf_samples": prepared["n_samples"],
                     "n_voxels_observed": int(prepared["uniq"].shape[0]),
                     "wallclock_sec": t_route_ii},
        "comparison": {
            "ff_relative_reduction": float(rel_red),
            "fo_multiple_vs_route_i": float(fo_mult),
            "ff_paired_bootstrap_mean_i_minus_ii": ff_ci,
            "meets_ff_bar": bool(rel_red >= REL_FF_REDUCTION_BAR),
            "meets_fo_bar": bool(fo_mult <= FO_MULTIPLE_BAR),
            "ci_excludes_zero_toward_reduction": bool(ff_ci["ci_lo"] > 0.0),
        },
    }
    arrays = {
        "occ_route_i": occ_i, "occ_route_ii": occ_ii,
        "free_at_gt_occ_route_i": conf_i["free_at_gt_occ"],
        "free_at_gt_occ_route_ii": conf_ii["free_at_gt_occ"],
        "occ_at_gt_free_route_i": conf_i["occ_at_gt_free"],
        "occ_at_gt_free_route_ii": conf_ii["occ_at_gt_free"],
        "d2_verdicts_route_i": v_i, "d2_verdicts_route_ii": v_ii,
        "d2_verdicts_gt": gt_verdicts,
    }
    return record, arrays, samples


def calibrate(prepared, grid, gt_occ, conf_i):
    """Sanctioned one-time grid on toy clean30k. Objective (pre-registered):
    minimize false_free_rate s.t. false_occupied_rate <= 2x route-(i)
    false_occupied_rate; tie-break lower false_occupied_rate, then grid order."""
    fo_cap = FO_MULTIPLE_BAR * conf_i["false_occupied_rate"]
    table = []
    for tau_t, iso, mv in itertools.product(TAU_T_GRID, ISO_GRID, MIN_VIEWS_GRID):
        occ = tsdf_occupancy(prepared, tuple(int(s) for s in grid.shape),
                             tau_t, iso, mv)
        c = d1_confusion(gt_occ, occ)
        table.append({
            "tau_t": tau_t, "iso": iso, "min_views": mv,
            "false_free_rate": c["false_free_rate"],
            "false_occupied_rate": c["false_occupied_rate"],
            "n_recon_occupied": c["n_recon_occupied"],
            "feasible_fo_leq_2x_route_i": bool(c["false_occupied_rate"] <= fo_cap),
        })
        print(f"  [cal] tau_t={tau_t:.2f} iso={iso:.2f} mv={mv} -> "
              f"ff={c['false_free_rate']:.4f} fo={c['false_occupied_rate']:.4f} "
              f"{'FEASIBLE' if c['false_occupied_rate'] <= fo_cap else 'infeasible'}")
    feasible = [r for r in table if r["feasible_fo_leq_2x_route_i"]]
    pool = feasible if feasible else table
    best = min(pool, key=lambda r: (r["false_free_rate"], r["false_occupied_rate"]))
    frozen = {"tau_t": best["tau_t"], "iso": best["iso"],
              "min_views": best["min_views"]}
    return table, frozen, bool(feasible), fo_cap


def _load_eval_crosscheck(path):
    try:
        with open(path) as f:
            ds = json.load(f).get("downstream", {})
        out = {}
        for k in ("d1", "d2"):
            sub = ds.get(k)
            if isinstance(sub, dict) and sub:
                out[k] = {kk: sub.get(kk) for kk in
                          ("false_free_rate", "false_occupied_rate",
                           "agreement_rate", "unsafe_disagreement_rate")
                          if kk in sub}
        return out if out else {"unavailable": "no d1/d2 in eval row"}
    except Exception as exc:
        return {"unavailable": str(exc)}


def run_study(out_root, gpu):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from tools.gems.eval_context import build_eval_context

    os.makedirs(out_root, exist_ok=True)
    t_study = time.time()
    scene_cache = {}
    summary = {
        "goal": "LEDGER GOAL #R-02 (Stage-1R R3.a)",
        "pre_registered_hypothesis": (
            "route (ii) TSDF fusion of train-view rendered median depths "
            "reduces toy d1 false_free_rate by >=50% relative vs route (i) "
            "triangle voxelization, at <=2x false-occupied, params calibrated "
            "ONCE on toy clean30k then FROZEN; PASS iff all 3 toy models meet "
            "the bar with paired CI excl. 0; courtyard = frozen-params "
            "transfer verdict"),
        "constants": {"voxel_m": VOXEL_M, "n_traj": N_TRAJ, "seed": SEED,
                      "alpha_min": ALPHA_MIN, "tau_t_grid": TAU_T_GRID,
                      "iso_grid": ISO_GRID, "min_views_grid": MIN_VIEWS_GRID},
        "models": {}, "caveats": [],
    }

    # ---- calibration on toy clean30k (must run first) ----
    cal_scene, cal_label = CALIBRATION_MODEL
    cal_entry = next(m for m in MODELS if (m[0], m[1]) == CALIBRATION_MODEL)
    spec, grid, gt_occ, trajs, gt_verdicts = _scene_assets(cal_scene)
    scene_cache[cal_scene] = (spec, grid, gt_occ, trajs, gt_verdicts)

    print(f"[r3a] calibration model: {cal_scene}/{cal_label}")
    ctx = build_eval_context(cal_entry[2], spec)
    finite = ctx.finite_faces_mask()
    occ_i_cal = route_i_occupancy(
        grid, ctx.vertices().cpu().numpy(),
        ctx.faces()[finite].cpu().numpy())
    conf_i_cal = d1_confusion(gt_occ, occ_i_cal)
    print(f"[r3a] route(i) toy clean: ff={conf_i_cal['false_free_rate']:.4f} "
          f"fo={conf_i_cal['false_occupied_rate']:.4f}")

    print("[r3a] collecting TSDF samples (train views only, D4) ...")
    cal_samples = collect_tsdf_samples(ctx, grid)
    cal_prepared = prepare_sorted_samples(cal_samples)
    print(f"[r3a] {cal_prepared['n_samples']:,} samples from "
          f"{cal_prepared['n_views']} train views")

    table, frozen, feasible_found, fo_cap = calibrate(
        cal_prepared, grid, gt_occ, conf_i_cal)
    calibration = {
        "calibration_model": f"{cal_scene}/{cal_label}",
        "route_i_false_free_rate": conf_i_cal["false_free_rate"],
        "route_i_false_occupied_rate": conf_i_cal["false_occupied_rate"],
        "fo_cap_2x_route_i": fo_cap,
        "table": table,
        "any_feasible": feasible_found,
        "frozen_params": frozen,
        "objective": ("min false_free_rate s.t. false_occupied_rate <= "
                      "2x route-i (pre-registered); tie-break lower fo"),
    }
    with open(os.path.join(out_root, "calibration_table.json"), "w") as f:
        json.dump(_json_safe(calibration), f, indent=1)
    with open(os.path.join(out_root, "frozen_params.json"), "w") as f:
        json.dump(_json_safe({"frozen_params": frozen,
                              "calibrated_on": f"{cal_scene}/{cal_label}",
                              "any_feasible": feasible_found}), f, indent=1)
    print(f"[r3a] FROZEN params: {frozen} (feasible={feasible_found})")
    summary["calibration"] = calibration

    # ---- all five (scene x model) cells with FROZEN params ----
    for scene, label, ckpt, eval_row in MODELS:
        key = f"{scene}__{label}"
        print(f"[r3a] === {key} ===")
        if scene not in scene_cache:
            scene_cache[scene] = _scene_assets(scene)
        spec, grid, gt_occ, trajs, gt_verdicts = scene_cache[scene]
        if (scene, label) == CALIBRATION_MODEL:
            model_ctx, samples = ctx, cal_samples
        else:
            model_ctx = build_eval_context(ckpt, spec)
            samples = None
        record, arrays, _ = _model_routes(
            model_ctx, grid, gt_occ, trajs, gt_verdicts, frozen, samples)
        record["checkpoint"] = ckpt
        record["scene"] = scene
        record["label"] = label
        record["route_i_crosscheck_vs_eval_row"] = {
            "eval_row": eval_row, **_load_eval_crosscheck(eval_row)}
        mdir = os.path.join(out_root, key)
        os.makedirs(mdir, exist_ok=True)
        np.savez_compressed(os.path.join(mdir, "grids_and_per_sample.npz"),
                            **arrays)
        record["arrays_npz"] = os.path.join(mdir, "grids_and_per_sample.npz")
        with open(os.path.join(mdir, "routes_metrics.json"), "w") as f:
            json.dump(_json_safe(record), f, indent=1)
        summary["models"][key] = _json_safe(record)
        c = record["comparison"]
        print(f"[r3a] {key}: ff {record['route_i']['d1']['false_free_rate']:.4f}"
              f" -> {record['route_ii']['d1']['false_free_rate']:.4f} "
              f"(rel red {c['ff_relative_reduction']:.3f}), fo x"
              f"{c['fo_multiple_vs_route_i']:.2f}")
        if (scene, label) != CALIBRATION_MODEL:
            del model_ctx
        import torch
        torch.cuda.empty_cache()

    # ---- verdicts vs the pre-registered bar ----
    toy_keys = [f"{s}__{l}" for s, l, _, _ in MODELS if s == "toy_parking"]
    cy_keys = [f"{s}__{l}" for s, l, _, _ in MODELS if s == "courtyard"]

    def model_pass(k):
        c = summary["models"][k]["comparison"]
        return bool(c["meets_ff_bar"] and c["meets_fo_bar"]
                    and c["ci_excludes_zero_toward_reduction"])

    toy_pass = all(model_pass(k) for k in toy_keys)
    summary["verdict"] = {
        "toy_per_model_pass": {k: model_pass(k) for k in toy_keys},
        "toy_preregistered_bar_PASS": toy_pass,
        "courtyard_per_model_pass_frozen_params": {k: model_pass(k) for k in cy_keys},
        "note": ("bar (pre-registered): >=50% relative false-free reduction at "
                 "<=2x false-occupied, paired CI excl. 0, on ALL 3 toy models; "
                 "courtyard reported as frozen-params transfer outcome"),
    }
    summary["caveats"] = [
        "courtyard frame is only approximately gravity-aligned (mean camera "
        "up-vector dot >= 0.908, LEDGER GOAL #008 derivation); the frozen ROI "
        "z_band is identical for both routes, so the paired comparison is "
        "unaffected; absolute occupancy semantics inherit the approximation",
        "route-(ii) surf_depth is the 4x-supersampled MEDIAN depth with "
        "background subpixels contributing 0 (PROTOCOL 4.3): silhouette-edge "
        "depths are diluted toward 0; the alpha >= 0.5 gate removes most such "
        "pixels but edge voxels can still receive biased-near samples",
        "route-(ii) marks unobserved voxels FREE (same semantics as route-(i) "
        "unmarked voxels); with min_views >= 2 sparsely-seen true surface "
        "becomes false-free rather than occupied",
        "courtyard GT = laser-scan points: voxels the scanner never sampled "
        "count as GT-free for BOTH routes (false_occupied inflation affects "
        "the routes equally)",
        "SS3DM excluded for now: B0 baselines still training (R3.a directive)",
        "D4: route (ii) consumes TRAIN views only; no test data anywhere",
    ]
    summary["wallclock_sec_total"] = time.time() - t_study
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(_json_safe(summary), f, indent=1)
    print(f"[r3a] wrote {os.path.join(out_root, 'summary.json')} "
          f"({summary['wallclock_sec_total']:.0f}s total)")
    return summary


# ---------------------------------------------------------------------------
# Self-test (numpy only, no GPU / renderer)
# ---------------------------------------------------------------------------

def selftest():
    # -- tsdf_occupancy aggregation semantics --
    # voxel 0: samples [-0.01, 0.01, 0.02] -> median 0.01 -> occupied (iso>0.01)
    # voxel 1: samples [0.5, 0.6], tau 0.3 -> clamped [0.3,0.3] -> median 0.3 -> free
    # voxel 2: single sample -0.5 -> below -tau -> no valid obs -> free
    # voxel 3: one sample 0.0 -> occupied only if min_views <= 1
    # voxel 4: even count [-0.2, 0.04] -> median of clamped = (-0.2+0.04)/2 = -0.08
    vox = np.array([0, 0, 0, 1, 1, 2, 3, 4, 4], dtype=np.int32)
    sdf = np.array([-0.01, 0.01, 0.02, 0.5, 0.6, -0.5, 0.0, -0.2, 0.04],
                   dtype=np.float32)
    prepared = prepare_sorted_samples(
        {"vox_ids": vox, "sdf": sdf, "n_voxels": 6, "n_views": 3,
         "per_view_n_samples": []})
    shape = (6, 1, 1)

    occ = tsdf_occupancy(prepared, shape, tau_t=0.3, iso=0.05, min_views=2)
    got = occ.ravel()
    assert got.tolist() == [True, False, False, False, False, False], got

    occ = tsdf_occupancy(prepared, shape, tau_t=0.3, iso=0.10, min_views=1)
    got = occ.ravel()
    # voxel 3 now passes min_views; voxel 4 median -0.08, |.| < 0.10 -> occupied
    assert got.tolist() == [True, False, False, True, True, False], got

    # occlusion-discard + median-of-clamped analytic case: voxel with samples
    # [-0.25, -0.05, 0.4] at tau 0.2 -> the -0.25 sample is occluded
    # (sdf <= -tau, discarded); valid = [-0.05, 0.4] -> clamped [-0.05, 0.2]
    # -> even-count median = 0.075.
    prepared2 = prepare_sorted_samples(
        {"vox_ids": np.zeros(3, np.int32),
         "sdf": np.array([-0.25, -0.05, 0.4], np.float32),
         "n_voxels": 1, "n_views": 3, "per_view_n_samples": []})
    occ = tsdf_occupancy(prepared2, (1, 1, 1), tau_t=0.2, iso=0.08, min_views=1)
    assert bool(occ.ravel()[0]) is True   # |0.075| < 0.08
    occ = tsdf_occupancy(prepared2, (1, 1, 1), tau_t=0.2, iso=0.06, min_views=1)
    assert bool(occ.ravel()[0]) is False  # |0.075| >= 0.06
    occ = tsdf_occupancy(prepared2, (1, 1, 1), tau_t=0.2, iso=0.08, min_views=3)
    assert bool(occ.ravel()[0]) is False  # only 2 valid obs < min_views=3
    # at tau 0.3 all 3 samples are valid: clamped [-0.25, -0.05, 0.3] ->
    # median -0.05 (odd count picks the middle order statistic)
    occ = tsdf_occupancy(prepared2, (1, 1, 1), tau_t=0.3, iso=0.06, min_views=3)
    assert bool(occ.ravel()[0]) is True   # |-0.05| < 0.06, 3 obs
    occ = tsdf_occupancy(prepared2, (1, 1, 1), tau_t=0.3, iso=0.04, min_views=3)
    assert bool(occ.ravel()[0]) is False  # |-0.05| >= 0.04

    # -- d1_confusion against the reference formulas --
    gt = np.zeros((2, 2, 1), dtype=bool)
    gt[0, 0, 0] = True
    gt[0, 1, 0] = True
    rec = np.zeros((2, 2, 1), dtype=bool)
    rec[0, 0, 0] = True   # hit
    rec[1, 0, 0] = True   # false occupied
    c = d1_confusion(gt, rec)
    assert c["false_free_rate"] == 0.5 and c["false_occupied_rate"] == 0.5, c
    assert c["free_at_gt_occ"].shape[0] == 2
    assert c["occ_at_gt_free"].shape[0] == 2

    # -- toy-scale synthetic end-to-end consistency of routes machinery:
    # a flat plane mesh voxelized by route (i) must match a TSDF whose samples
    # are the exact plane sdf at voxel centers (analytic "perfect depth").
    roi = {"min": [0.0, 0.0], "max": [2.0, 2.0], "z_band": [0.0, 0.5]}
    grid = build_grid(roi)
    v = np.array([[-5, -5, 0.25], [5, -5, 0.25], [5, 5, 0.25], [-5, 5, 0.25]],
                 dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    occ_i = route_i_occupancy(grid, v, f)
    shape = tuple(int(s) for s in grid.shape)
    n_vox = int(np.prod(shape))
    ii, jj, kk = np.unravel_index(np.arange(n_vox), shape)
    zc = grid.origin[2] + (kk + 0.5) * grid.voxel
    sdf_analytic = (0.25 - zc).astype(np.float32)  # signed distance along "ray"
    prepared3 = prepare_sorted_samples(
        {"vox_ids": np.tile(np.arange(n_vox, dtype=np.int32), 2),
         "sdf": np.tile(sdf_analytic, 2), "n_voxels": n_vox, "n_views": 2,
         "per_view_n_samples": []})
    occ_ii = tsdf_occupancy(prepared3, shape, tau_t=0.3, iso=0.05, min_views=2)
    # plane z=0.25 sits exactly on the boundary between layers 2 (|sdf|=0.0)
    # ... voxel centers at z = 0.05,0.15,0.25,0.35,0.45 -> layer k=2 has
    # sdf 0.0 -> occupied; neighbors have |sdf| 0.1 >= iso -> free.
    assert occ_ii[:, :, 2].all() and not occ_ii[:, :, [0, 1, 3, 4]].any()
    assert occ_i[:, :, 2].all()          # route (i) marks the same layer
    conf = d1_confusion(occ_i, occ_ii)   # treating route (i) as "GT"
    assert conf["false_free_rate"] == 0.0
    print("occupancy_routes selftest PASSED")


def main():
    ap = argparse.ArgumentParser(description="GEMS R3.a occupancy routes study")
    ap.add_argument("--out", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run_study(args.out, args.gpu)


if __name__ == "__main__":
    main()
