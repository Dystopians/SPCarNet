#!/usr/bin/env python
"""GEMS Stage-1R R3.b — certified structural sub-mesh (LEDGER GOAL #R-06).

One-time GLOBAL triangle labeling per checkpoint (TRAIN evidence only, D4):

  support(t)   — # of the <=60 evenly spaced train views (the exact g3
                 support pass) where triangle t owns >= 1 valid rend_ids
                 pixel (gated by depth_full > 0).
  cons(t)      — # of those views where t is SUPPORTED and its center
                 (mean of its 3 vertices) projects in-frame & in-front
                 (repo-exact _project_points), rend_alpha >= 0.5,
                 surf_depth d_r finite > 0, and
                 |depth(center) - d_r| <= tau_d * d_r  (tau_d FROZEN 0.10).

  keep(t) = finite(t) AND support(t) >= k AND cons(t) >= m.

The kept set is the "collision-grade sub-mesh": consumed by route-i
voxelization (the exact d1 _rasterize_triangles) -> d1/d2 -> the R-03
planner harness UNCHANGED (same seed-0 100 problems per scene).

Framing guard (Stage-1R insert §R3): the labels are a ONE-TIME, GLOBAL,
TRAIN-EVIDENCE-ONLY artifact — one keep-set per checkpoint, frozen as npz;
nothing varies per view or per query at consumption. This is an
artifact-generation step, NOT a selector.

Sanctioned ONE-TIME calibration (toy clean30k ONLY, table logged):
k in {2,3,5} x m in {1,2}; objective (pre-registered): minimize planner
spurious-infeasibility s.t. collisions-per-100-plans <= toy GTREF floor + 1
and d1 false-occupied <= 1.5x raw route-i (structurally guaranteed:
kept faces are a subset, so sub-mesh occupancy is a subset of raw route-i
occupancy). Tie-break: lower collisions, then lower d1 false-free, then
grid order. FREEZE; apply unchanged to toy{clean,B50}, courtyard{clean,B50}
(+ toy B25 as an extra reported row, not part of the bar).

Verification before any sub-mesh planning (abort on mismatch):
  v1 — rebuilt GT occupancy reproduces the stored R3.a per-voxel indicators
       and stored seed-0 d2 GT verdicts bit-exact (verify_gt_grid, as R-03);
  v2 — re-planned GTREF per-problem records match the stored R-03 GTREF
       records exactly (found/reason/n_expansions/path_length/segments/
       switches) => the seed-0 problems are the identical paired set;
  v3 — voxelizing keep=ALL-finite-faces reproduces the stored occ_route_i
       bit-exact per cell (same checkpoint, same voxelizer).

Usage:
    python tools/gems/certified_submesh.py --gpu 1        # full study
    python tools/gems/certified_submesh.py --selftest     # numpy-only tests

Durable outputs: /data/peilincai/gems_stage1/analysis/r3b_submesh/
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# numpy/scipy-only modules — safe before CUDA_VISIBLE_DEVICES is set.
from tools.gems.downstream_metrics import (          # noqa: E402
    _VoxelGrid,
    _build_gt_occupancy,
    _collision_verdicts,
    _rasterize_triangles,
    _sample_trajectories,
)
from tools.gems.occupancy_routes import (            # noqa: E402
    MODELS as R3A_MODELS,
    build_gt_arg,
    d1_confusion,
)
from tools.gems.paired_bootstrap import paired_bootstrap_ci  # noqa: E402
from tools.gems import planner_loop as pl            # noqa: E402

# ---------------------------------------------------------------------------
# Frozen study constants (pre-registered in LEDGER GOAL #R-06)
# ---------------------------------------------------------------------------
VOXEL_M = 0.10            # PROTOCOL 4.4 d1 voxel
N_TRAJ = 200              # PROTOCOL 4.4 d2 trajectories
SEED = 0                  # PROTOCOL seed — UNCHANGED everywhere
ALPHA_MIN = 0.5           # same gate as g1/g2/route-ii
MAX_LABEL_VIEWS = 60      # the g3 support-pass view budget

TAU_D = 0.10              # FROZEN relative depth-consistency band
K_GRID = (2, 3, 5)        # sanctioned calibration: support threshold
M_GRID = (1, 2)           # sanctioned calibration: consistency threshold

COLL_CAP_OVER_GTREF = 1.0     # collisions cap = GTREF floor + 1 (per 100)
FO_MULT_CAP = 1.5             # d1 false-occupied cap vs raw route-i
FF_REL_WORSE_BAR = 0.10       # prediction: ff not >10% rel worse than raw
FOUND_BAR = {"toy_parking": 0.80, "courtyard": 0.30}   # prediction arms

OUT_ROOT_DEFAULT = "/data/peilincai/gems_stage1/analysis/r3b_submesh"
R3C_ROOT = "/data/peilincai/gems_stage1/analysis/r3c_planner"

# Bar cells (pre-registered) + extra reported row (not part of the bar).
BAR_CELLS = [
    ("toy_parking", "clean30k"),
    ("toy_parking", "B50_importance_ft_e1v2_40000"),
    ("courtyard", "clean30k"),
    ("courtyard", "B50_importance_ft_e1v2_40000"),
]
EXTRA_CELLS = [("toy_parking", "B25_importance_ft_e1v2_40000")]
CALIBRATION_CELL = ("toy_parking", "clean30k")
# Processing order: calibration cell FIRST (params must freeze before any
# other cell is labeled/planned with them).
CELL_ORDER = [
    ("toy_parking", "clean30k"),
    ("toy_parking", "B50_importance_ft_e1v2_40000"),
    ("toy_parking", "B25_importance_ft_e1v2_40000"),
    ("courtyard", "clean30k"),
    ("courtyard", "B50_importance_ft_e1v2_40000"),
]


def _ckpt_for(scene, label):
    for s, l, ckpt, _ in R3A_MODELS:
        if (s, l) == (scene, label):
            return ckpt
    raise KeyError(f"no R3.a model entry for {scene}/{label}")


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# Labeling pass (GPU; train views only — D4)
# ---------------------------------------------------------------------------

def label_checkpoint(ctx, tau_d=TAU_D, max_views=MAX_LABEL_VIEWS,
                     alpha_min=ALPHA_MIN):
    """One pass over the g3 support view set. Returns global per-triangle
    counts: support (g3 definition) and cons (supported AND center-depth
    consistent within tau_d relative of the model's own median depth)."""
    import torch
    from tools.gems.geometry_metrics import _project_points

    faces = ctx.faces()                       # [T,3] long cuda
    verts = ctx.vertices()                    # [V,3] cuda
    n_tri = int(faces.shape[0])
    if n_tri >= 2 ** 24:
        raise ValueError("rend_ids float32 cannot represent ids >= 2^24")
    centers = verts[faces].mean(dim=1).float().contiguous()

    n_views = min(int(max_views), len(ctx.train_cams))
    view_indices = np.unique(
        np.linspace(0, len(ctx.train_cams) - 1, n_views).round().astype(int))
    support = torch.zeros(n_tri, dtype=torch.int32, device="cuda")
    cons = torch.zeros(n_tri, dtype=torch.int32, device="cuda")

    with torch.no_grad():
        for vi in view_indices:
            cam = ctx.train_cams[int(vi)]
            pkg = ctx.render_view(cam)

            # -- support: exact g3 _support_counts semantics --
            ids = pkg["rend_ids"].detach().reshape(-1)
            valid = (ids >= 0) & (ids < n_tri)
            depth_full = pkg.get("depth_full")
            if depth_full is not None:
                valid &= depth_full.detach().reshape(-1) > 0
            sup_v = torch.zeros(n_tri, dtype=torch.bool, device="cuda")
            sup_v[torch.unique(ids[valid].round().long())] = True

            # -- center depth consistency vs the model's own median depth --
            depth_c, px, py, in_front = _project_points(centers, cam)
            W, H = int(cam.image_width), int(cam.image_height)
            ok = in_front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            d_map = pkg["surf_depth"].detach()[0]     # [H, W]
            a_map = pkg["rend_alpha"].detach()[0]     # [H, W]
            d_r = d_map[py[ok], px[ok]]
            a_r = a_map[py[ok], px[ok]]
            good = ((a_r >= alpha_min) & torch.isfinite(d_r) & (d_r > 0.0)
                    & ((depth_c[ok] - d_r).abs() <= tau_d * d_r))
            cons_v = torch.zeros(n_tri, dtype=torch.bool, device="cuda")
            idxs = torch.nonzero(ok, as_tuple=False).squeeze(1)
            cons_v[idxs[good]] = True

            support += sup_v.to(torch.int32)
            cons += (sup_v & cons_v).to(torch.int32)
            del (pkg, ids, valid, sup_v, cons_v, depth_c, px, py, in_front,
                 ok, d_map, a_map, d_r, a_r, good, idxs)
            torch.cuda.empty_cache()

    return {
        "support": support.cpu().numpy().astype(np.int16),
        "cons": cons.cpu().numpy().astype(np.int16),
        "view_indices": view_indices.astype(np.int64),
        "n_views": int(view_indices.shape[0]),
        "n_triangles": n_tri,
        "tau_d": float(tau_d),
    }


def keep_mask(support, cons, finite, k, m):
    """keep(t) = finite AND support >= k AND cons >= m (original face order)."""
    return (np.asarray(finite, dtype=bool)
            & (np.asarray(support) >= int(k))
            & (np.asarray(cons) >= int(m)))


def submesh_occupancy(grid, verts_np, faces_np, keep):
    occ = grid.new_occupancy()
    _rasterize_triangles(grid, occ, verts_np, faces_np[keep])
    return occ


# ---------------------------------------------------------------------------
# v2 — GTREF replan must reproduce the stored R3.c records exactly
# ---------------------------------------------------------------------------

_V2_FIELDS = ("found", "reason", "n_expansions", "n_segments", "n_switches")


def compare_gtref_records(per_problem, stored_per_problem):
    """Exact per-problem comparison on the deterministic fields (wall_s is
    time and excluded; path_length_m compared to 1e-9)."""
    mismatches = []
    if len(per_problem) != len(stored_per_problem):
        return [f"count {len(per_problem)} != {len(stored_per_problem)}"]
    for new, old in zip(per_problem, stored_per_problem):
        for f in _V2_FIELDS:
            if new.get(f) != old.get(f):
                mismatches.append(
                    f"problem {new['problem']}: {f} {new.get(f)!r} != "
                    f"{old.get(f)!r}")
        a, b = new.get("path_length_m"), old.get("path_length_m")
        if (a is None) != (b is None) or \
                (a is not None and abs(a - b) > 1e-9):
            mismatches.append(
                f"problem {new['problem']}: path_length_m {a} != {b}")
    return mismatches


def load_r3c_per_problem(cell_key):
    path = os.path.join(R3C_ROOT, cell_key, "cell_metrics.json")
    with open(path) as f:
        return json.load(f)["per_problem"], path


# ---------------------------------------------------------------------------
# Calibration (sanctioned one-time; toy clean30k ONLY)
# ---------------------------------------------------------------------------

def calibrate(labels, finite, verts_np, faces_np, grid, gt_occ, conf_raw,
              ptab, gt_band, problems, gtref_floor):
    """k x m grid on toy clean30k. Objective (pre-registered): minimize
    planner spurious-infeasibility s.t. collisions-per-100 <= gtref_floor + 1
    and d1 false-occupied <= 1.5x raw route-i. Tie-break: lower collisions,
    then lower d1 false-free, then grid order."""
    fo_cap = FO_MULT_CAP * conf_raw["false_occupied_rate"]
    coll_cap = gtref_floor + COLL_CAP_OVER_GTREF
    n_finite = int(np.asarray(finite, dtype=bool).sum())
    table = []
    for gi, (k, m) in enumerate(itertools.product(K_GRID, M_GRID)):
        keep = keep_mask(labels["support"], labels["cons"], finite, k, m)
        occ = submesh_occupancy(grid, verts_np, faces_np, keep)
        conf = d1_confusion(gt_occ, occ)
        band = pl.footprint_layer(occ, grid)
        maps = pl.GridMaps(band, ptab)
        goal_cells = [p["goal"][0] * maps.ny + p["goal"][1] for p in problems]
        dist_maps = maps.dijkstra_from(goal_cells)
        metrics, _ = pl.plan_cell(f"cal_k{k}_m{m}", maps, gt_band, problems,
                                  ptab, dist_maps)
        coll = metrics["collisions_per_100_plans"]
        feas_coll = (coll is None) or (coll <= coll_cap + 1e-12)
        feas_fo = conf["false_occupied_rate"] <= fo_cap + 1e-12
        row = {
            "grid_index": gi, "k": k, "m": m,
            "kept_fraction_of_finite": float(keep.sum()) / n_finite,
            "n_kept": int(keep.sum()),
            "false_free_rate": conf["false_free_rate"],
            "false_occupied_rate": conf["false_occupied_rate"],
            "found_rate": metrics["found_rate"],
            "spurious_infeasibility_rate": metrics["spurious_infeasibility_rate"],
            "collisions_per_100_plans": coll,
            "n_gt_collisions": metrics["n_gt_collisions"],
            "feasible_coll_leq_gtref_plus_1": bool(feas_coll),
            "feasible_fo_leq_1p5x_raw": bool(feas_fo),
            "feasible": bool(feas_coll and feas_fo),
        }
        table.append(row)
        print(f"  [cal] k={k} m={m}: kept {row['kept_fraction_of_finite']:.3f}"
              f" found {metrics['found_rate']:.2f} coll/100 {coll} "
              f"ff {conf['false_free_rate']:.4f} fo "
              f"{conf['false_occupied_rate']:.4f} "
              f"{'FEASIBLE' if row['feasible'] else 'infeasible'}")

    feasible = [r for r in table if r["feasible"]]
    pool = feasible if feasible else table

    def sort_key(r):
        coll = r["collisions_per_100_plans"]
        return (r["spurious_infeasibility_rate"],
                coll if coll is not None else float("inf"),
                r["false_free_rate"], r["grid_index"])

    best = min(pool, key=sort_key)
    frozen = {"k": best["k"], "m": best["m"], "tau_d": TAU_D}
    return table, frozen, bool(feasible), {"fo_cap": fo_cap,
                                           "coll_cap": coll_cap}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def footprint_panel(png_path, scene, grid, panels):
    """Side-by-side occupancy footprints: each entry of `panels` is
    (subtitle, band2d, lethal2d, gt_band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    nx, ny = panels[0][1].shape
    x0, y0 = grid.origin[0], grid.origin[1]
    extent = [x0, x0 + nx * VOXEL_M, y0, y0 + ny * VOXEL_M]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(8 * len(panels), 8 * ny / nx))
    for ax, (subtitle, band, lethal, gt_band) in zip(np.atleast_1d(axes),
                                                     panels):
        img = np.ones((nx, ny, 3))
        for mask, color, alpha in [
                (lethal, pl._hex_rgb(pl.C_LETHAL), 1.0),
                (band, pl._hex_rgb(pl.C_OCC), 1.0),
                (gt_band, pl._hex_rgb(pl.C_GT), 0.55)]:
            img[mask] = (1 - alpha) * img[mask] + alpha * color
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower",
                  extent=extent, interpolation="nearest")
        ax.set_title(subtitle, fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    legend = [
        Patch(facecolor=pl.C_OCC, label="model grid (footprint layer)"),
        Patch(facecolor=pl.C_LETHAL, label="inflated lethal (ESDF <= 1.0 m)"),
        Patch(facecolor=pl.C_GT, alpha=0.55, label="GT footprint layer"),
    ]
    np.atleast_1d(axes)[-1].legend(handles=legend, loc="upper right",
                                   fontsize=7, framealpha=0.9)
    fig.suptitle(f"{scene}: raw route-i vs certified sub-mesh "
                 f"(GOAL #R-06)", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

def run_study(out_root, gpu):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from tools.gems.eval_context import build_eval_context
    from tools.gems.scenes import SCENES
    import torch

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "panels"), exist_ok=True)
    t_study = time.time()

    print("[r3b] building primitive table ...")
    ptab = pl.PrimitiveTable()

    summary = {
        "goal": "LEDGER GOAL #R-06 (Stage-1R R3.b certified structural sub-mesh)",
        "pre_registered_mechanism": (
            "one-time GLOBAL train-evidence-only labeling per checkpoint: "
            "keep(t) = finite AND support(t) >= k (g3 support pass, <=60 "
            "train views) AND cons(t) >= m where cons counts supporting "
            "views whose center depth is within tau_d=0.10 relative of the "
            "model's own median depth; k x m calibrated ONCE on toy clean30k "
            "(<=3x2 grid), FROZEN, applied unchanged elsewhere"),
        "pre_registered_prediction": (
            "certified sub-mesh restores planner feasibility (found-rate >= "
            "80/100 on toy bar cells, >= 30/100 on courtyard bar cells) at "
            "collisions-per-100-plans <= scene GTREF floor + 1, with d1 "
            "false-free not worse than raw route-i by more than 10% relative "
            "per bar cell; PASS iff all arms hold on all 4 bar cells"),
        "constants": {
            "voxel_m": VOXEL_M, "n_traj": N_TRAJ, "seed": SEED,
            "alpha_min": ALPHA_MIN, "max_label_views": MAX_LABEL_VIEWS,
            "tau_d_frozen": TAU_D, "k_grid": K_GRID, "m_grid": M_GRID,
            "coll_cap_over_gtref": COLL_CAP_OVER_GTREF,
            "fo_mult_cap": FO_MULT_CAP,
            "ff_rel_worse_bar": FF_REL_WORSE_BAR,
            "found_bar": FOUND_BAR,
            "bar_cells": [f"{s}__{l}" for s, l in BAR_CELLS],
            "extra_cells": [f"{s}__{l}" for s, l in EXTRA_CELLS],
        },
        "verification": {}, "scenes": {}, "cells": {}, "comparisons": {},
        "caveats": [],
    }

    # ---- per-scene assets + verification (v1, v2) ----
    scene_labels = {}
    for scene, label in CELL_ORDER:
        scene_labels.setdefault(scene, []).append(label)

    scenes = {}
    for scene, labels in scene_labels.items():
        print(f"[r3b] === scene {scene}: GT rebuild + verification ===")
        spec = SCENES[scene]
        grid = _VoxelGrid(spec.roi, VOXEL_M)
        gt_occ = _build_gt_occupancy(grid, build_gt_arg(spec))
        cells_r3a = {}
        for label in labels:
            arrs, path = pl.load_r3a_cell(scene, label)
            cells_r3a[label] = arrs
        ok, checks = pl.verify_gt_grid(scene, grid, gt_occ, cells_r3a)
        summary["verification"][f"{scene}__v1_gt_bit_exact"] = \
            {"all_bit_exact": ok, **checks}
        if not ok:
            raise RuntimeError(f"[v1] GT verification FAILED for {scene}: "
                               f"{checks}")
        print(f"[r3b]   v1 GT bit-exact: {ok} ({len(checks)} checks)")

        trajs = _sample_trajectories(np.random.default_rng(SEED), spec.roi,
                                     N_TRAJ)
        gt_verdicts = _collision_verdicts(trajs, grid, gt_occ)
        gt_band = pl.footprint_layer(gt_occ, grid)
        maps_gt = pl.GridMaps(gt_band, ptab)
        problems = pl.sample_problems(maps_gt, np.random.default_rng(SEED))
        goal_cells = [p["goal"][0] * maps_gt.ny + p["goal"][1]
                      for p in problems]
        dist_gt = maps_gt.dijkstra_from(goal_cells)
        gtref_metrics, gtref_pp = pl.plan_cell(
            f"{scene}__GTREF", maps_gt, gt_band, problems, ptab, dist_gt)
        stored_pp, stored_path = load_r3c_per_problem(f"{scene}__GTREF")
        mism = compare_gtref_records(
            [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in gtref_pp], stored_pp)
        summary["verification"][f"{scene}__v2_gtref_replan"] = {
            "identical": not mism, "stored": stored_path,
            "n_mismatches": len(mism), "mismatches": mism[:10],
        }
        if mism:
            raise RuntimeError(f"[v2] GTREF replan mismatch for {scene}: "
                               f"{mism[:5]}")
        gtref_floor = gtref_metrics["collisions_per_100_plans"]
        print(f"[r3b]   v2 GTREF replan identical to R3.c "
              f"(floor {gtref_floor}/100)")
        summary["scenes"][scene] = {
            "grid_shape": [maps_gt.nx, maps_gt.ny],
            "gtref_collisions_per_100": gtref_floor,
            "coll_cap": gtref_floor + COLL_CAP_OVER_GTREF,
            "n_problems": len(problems),
        }
        scenes[scene] = {
            "spec": spec, "grid": grid, "gt_occ": gt_occ, "trajs": trajs,
            "gt_verdicts": gt_verdicts, "gt_band": gt_band,
            "maps_gt": maps_gt, "problems": problems,
            "gtref_pp": gtref_pp, "gtref_floor": gtref_floor,
            "cells_r3a": cells_r3a,
        }

    # ---- cells: label (GPU), v3 check, calibrate-once, frozen application --
    frozen = None
    calibration = None
    cell_records = {}
    for scene, label in CELL_ORDER:
        key = f"{scene}__{label}"
        sc = scenes[scene]
        grid, gt_occ = sc["grid"], sc["gt_occ"]
        ckpt = _ckpt_for(scene, label)
        print(f"[r3b] === {key}: labeling (train views only, D4) ===")
        t0 = time.time()
        ctx = build_eval_context(ckpt, sc["spec"])
        finite = ctx.finite_faces_mask().detach().cpu().numpy().astype(bool)
        verts_np = ctx.vertices().cpu().numpy()
        faces_np = ctx.faces().cpu().numpy()
        labels_d = label_checkpoint(ctx)
        t_label = time.time() - t0
        del ctx
        torch.cuda.empty_cache()
        print(f"[r3b]   labeled {labels_d['n_triangles']:,} tris over "
              f"{labels_d['n_views']} views in {t_label:.0f}s")

        # v3: keep=ALL finite faces must reproduce the stored occ_route_i.
        occ_full = submesh_occupancy(grid, verts_np, faces_np, finite)
        v3_ok = bool(np.array_equal(occ_full,
                                    sc["cells_r3a"][label]["occ_route_i"]))
        summary["verification"][f"{key}__v3_full_faces_bit_exact"] = v3_ok
        if not v3_ok:
            raise RuntimeError(f"[v3] full-finite voxelization != stored "
                               f"occ_route_i for {key}")
        conf_raw = d1_confusion(gt_occ, occ_full)
        print(f"[r3b]   v3 bit-exact vs stored occ_route_i: {v3_ok}")

        # ---- sanctioned one-time calibration on toy clean30k ----
        if (scene, label) == CALIBRATION_CELL:
            print("[r3b] calibration grid (toy clean30k ONLY) ...")
            table, frozen, any_feasible, caps = calibrate(
                labels_d, finite, verts_np, faces_np, grid, gt_occ, conf_raw,
                ptab, sc["gt_band"], sc["problems"], sc["gtref_floor"])
            calibration = {
                "calibration_cell": key,
                "raw_route_i_false_free_rate": conf_raw["false_free_rate"],
                "raw_route_i_false_occupied_rate":
                    conf_raw["false_occupied_rate"],
                **caps, "gtref_floor": sc["gtref_floor"],
                "table": table, "any_feasible": any_feasible,
                "frozen_params": frozen,
                "objective": ("min spurious-infeasibility s.t. coll/100 <= "
                              "GTREF+1 and fo <= 1.5x raw route-i "
                              "(pre-registered); tie-break lower coll, then "
                              "lower ff, then grid order"),
            }
            with open(os.path.join(out_root, "calibration_table.json"),
                      "w") as f:
                json.dump(_json_safe(calibration), f, indent=1)
            with open(os.path.join(out_root, "frozen_params.json"), "w") as f:
                json.dump(_json_safe(
                    {"frozen_params": frozen, "calibrated_on": key,
                     "any_feasible": any_feasible,
                     "tau_d_frozen_at_preregistration": TAU_D}), f, indent=1)
            summary["calibration"] = calibration
            print(f"[r3b] FROZEN params: {frozen} "
                  f"(feasible={any_feasible})")
        assert frozen is not None, "calibration cell must run first"

        # ---- frozen application ----
        keep = keep_mask(labels_d["support"], labels_d["cons"], finite,
                         frozen["k"], frozen["m"])
        n_finite = int(finite.sum())
        occ_sub = submesh_occupancy(grid, verts_np, faces_np, keep)
        conf_sub = d1_confusion(gt_occ, occ_sub)

        # d1 paired ff CI vs stored raw route-i (units = GT-occupied voxels).
        raw_ff = sc["cells_r3a"][label]["free_at_gt_occ_route_i"]
        ff_ci = paired_bootstrap_ci(
            conf_sub["free_at_gt_occ"].astype(np.float64),
            raw_ff.astype(np.float64))
        ff_raw_rate = float(raw_ff.mean())
        ff_rel_change = ((conf_sub["false_free_rate"] - ff_raw_rate)
                         / ff_raw_rate if ff_raw_rate > 0 else float("nan"))

        # d2 on the sub-mesh grid (same 200 seed-0 trajectories).
        v_sub = _collision_verdicts(sc["trajs"], grid, occ_sub)
        gtv = sc["gt_verdicts"]
        n_gt_coll = int(gtv.sum())
        d2 = {
            "agreement_rate": float((v_sub == gtv).mean()),
            "unsafe_disagreement_rate": (float((~v_sub[gtv]).mean())
                                         if n_gt_coll else float("nan")),
            "n_recon_collision": int(v_sub.sum()),
            "n_gt_collision": n_gt_coll, "n_traj": len(gtv), "seed": SEED,
        }

        # planner on the sub-mesh grid (same seed-0 problems — paired).
        band_sub = pl.footprint_layer(occ_sub, grid)
        maps_sub = pl.GridMaps(band_sub, ptab)
        goal_cells = [p["goal"][0] * maps_sub.ny + p["goal"][1]
                      for p in sc["problems"]]
        dist_maps = maps_sub.dijkstra_from(goal_cells)
        metrics, per_problem = pl.plan_cell(
            f"{key}__submesh", maps_sub, sc["gt_band"], sc["problems"],
            ptab, dist_maps)
        metrics["path_length_inflation_vs_gtref"] = \
            pl.path_length_inflation(per_problem, sc["gtref_pp"])

        # paired comparisons vs the stored R3.c raw route-i cell.
        raw_pp, raw_path = load_r3c_per_problem(f"{key}__route_i")
        comp = {
            "vs_raw_route_i_cell": raw_path,
            "found_submesh_minus_raw": _json_safe(
                pl.paired_found_ci(per_problem, raw_pp)),
            "collisions_submesh_minus_raw_common_found": _json_safe(
                pl.paired_collision_ci(per_problem, raw_pp)),
        }

        record = {
            "cell": key, "checkpoint": ckpt,
            "is_bar_cell": (scene, label) in BAR_CELLS,
            "labeling": {
                "n_triangles": labels_d["n_triangles"],
                "n_finite": n_finite,
                "n_views": labels_d["n_views"],
                "tau_d": labels_d["tau_d"],
                "wallclock_sec": t_label,
                "support_histogram_percentiles": {
                    str(q): float(np.percentile(labels_d["support"], q))
                    for q in (10, 25, 50, 75, 90)},
            },
            "frozen_params": frozen,
            "kept": {
                "n_kept": int(keep.sum()),
                "kept_fraction_of_finite": float(keep.sum()) / n_finite,
                "kept_fraction_of_total": float(keep.mean()),
            },
            "d1_submesh": _json_safe(conf_sub),
            "d1_raw_route_i": {"false_free_rate": ff_raw_rate,
                               "false_occupied_rate": float(
                                   sc["cells_r3a"][label]
                                   ["occ_at_gt_free_route_i"].mean())},
            "d1_ff_paired_ci_submesh_minus_raw": ff_ci,
            "d1_ff_relative_change_vs_raw": float(ff_rel_change),
            "d2_submesh": d2,
            "planner": _json_safe(metrics),
            "comparisons": comp,
        }
        cdir = os.path.join(out_root, key)
        os.makedirs(cdir, exist_ok=True)
        np.savez_compressed(
            os.path.join(cdir, "labels.npz"),
            support=labels_d["support"], cons=labels_d["cons"],
            finite=finite, keep_frozen=keep,
            view_indices=labels_d["view_indices"],
            tau_d=np.float64(TAU_D), k=np.int64(frozen["k"]),
            m=np.int64(frozen["m"]))
        np.savez_compressed(
            os.path.join(cdir, "submesh_grids.npz"),
            occ_submesh=occ_sub,
            free_at_gt_occ_submesh=conf_sub["free_at_gt_occ"],
            occ_at_gt_free_submesh=conf_sub["occ_at_gt_free"],
            d2_verdicts_submesh=v_sub, d2_verdicts_gt=gtv)
        clean_pp = [{k2: v2 for k2, v2 in r.items()
                     if not k2.startswith("_")} for r in per_problem]
        with open(os.path.join(cdir, "cell_metrics.json"), "w") as f:
            json.dump(_json_safe({**record, "per_problem": clean_pp}), f,
                      indent=1)
        summary["cells"][key] = _json_safe(record)
        # non-serializable working objects (panels only) are attached AFTER
        # the summary snapshot — they must never reach json.dump
        record["_per_problem"] = per_problem
        record["_maps_sub"] = maps_sub
        record["_band_sub"] = band_sub
        cell_records[key] = record
        print(f"[r3b]   {key}: kept {record['kept']['kept_fraction_of_finite']:.3f} "
              f"| found {metrics['plans_found']}/{metrics['n_problems']} "
              f"| coll/100 {metrics['collisions_per_100_plans']} "
              f"| ff {conf_sub['false_free_rate']:.4f} "
              f"(raw {ff_raw_rate:.4f}, rel {ff_rel_change:+.3f}) "
              f"| fo {conf_sub['false_occupied_rate']:.4f}")

    # ---- panels: 2 per scene (before/after footprint + one planned path) --
    panel_paths = {}
    for scene in scenes:
        sc = scenes[scene]
        b50_label = next(l for s, l in CELL_ORDER
                         if s == scene and l.startswith("B50"))
        key = f"{scene}__{b50_label}"
        rec = cell_records[key]
        band_raw = pl.footprint_layer(sc["cells_r3a"][b50_label]
                                      ["occ_route_i"], sc["grid"])
        maps_raw = pl.GridMaps(band_raw, ptab)
        raw_metrics = json.load(open(os.path.join(
            R3C_ROOT, f"{key}__route_i", "cell_metrics.json")))["metrics"]
        png1 = os.path.join(out_root, "panels",
                            f"{scene}__footprint_before_after.png")
        footprint_panel(png1, scene, sc["grid"], [
            (f"BEFORE: raw route-i ({b50_label})\nlethal "
             f"{maps_raw.lethal.mean():.1%}, found "
             f"{raw_metrics['plans_found']}/100", band_raw, maps_raw.lethal,
             sc["gt_band"]),
            (f"AFTER: certified sub-mesh (k={frozen['k']}, m={frozen['m']}, "
             f"tau_d={TAU_D})\nkept {rec['kept']['kept_fraction_of_finite']:.1%}"
             f" of finite tris, lethal {rec['_maps_sub'].lethal.mean():.1%}, "
             f"found {rec['planner']['plans_found']}/100",
             rec["_band_sub"], rec["_maps_sub"].lethal, sc["gt_band"]),
        ])
        panel_paths[f"{scene}__footprint_before_after"] = png1
        print(f"[r3b]   panel: {png1}")

        # one planned path on the certified B50 grid
        ex = None
        for pred in (lambda r: r["found"] and not r["gt_collision"],
                     lambda r: r["found"],
                     lambda r: True):
            ex = next((r for r in rec["_per_problem"] if pred(r)), None)
            if ex is not None:
                break
        png2 = os.path.join(out_root, "panels",
                            f"{scene}__submesh_planned_path.png")
        prob = sc["problems"][ex["problem"]]
        sweep = None
        if ex["found"]:
            sweep = pl.gt_sweep_collision(ex["_segments"], prob["start"],
                                          ptab, sc["gt_band"],
                                          collect_mask=True)
        plen = (f"{ex['path_length_m']:.1f} m"
                if ex.get("path_length_m") is not None else "n/a")
        pl.draw_panel(png2,
                      f"{key}__submesh | problem {ex['problem']} | "
                      f"reason={ex['reason']} len={plen}",
                      sc["grid"], rec["_maps_sub"], sc["gt_band"], prob, ex,
                      sweep, ptab)
        panel_paths[f"{scene}__submesh_planned_path"] = png2
        print(f"[r3b]   panel: {png2}")
    summary["panels"] = panel_paths

    # ---- verdict vs the pre-registered prediction ----
    verdict = {"per_bar_cell": {}, "note": (
        "prediction (frozen at pre-registration): found-rate >= 0.80 (toy) / "
        ">= 0.30 (courtyard), collisions-per-100 <= GTREF floor + 1, d1 "
        "false-free <= 1.10x raw route-i, on ALL 4 bar cells; extra cells "
        "reported, not part of the bar")}
    all_ok = True
    for scene, label in BAR_CELLS:
        key = f"{scene}__{label}"
        rec = summary["cells"][key]
        m = rec["planner"]
        coll = m["collisions_per_100_plans"]
        cap = summary["scenes"][scene]["coll_cap"]
        found_ok = m["found_rate"] >= FOUND_BAR[scene]
        coll_ok = (coll is not None) and (coll <= cap + 1e-12)
        ff_ok = rec["d1_ff_relative_change_vs_raw"] <= FF_REL_WORSE_BAR
        cell_ok = bool(found_ok and coll_ok and ff_ok)
        all_ok &= cell_ok
        verdict["per_bar_cell"][key] = {
            "found_rate": m["found_rate"], "found_ok": bool(found_ok),
            "collisions_per_100": coll, "coll_cap": cap,
            "coll_ok": bool(coll_ok),
            "ff_relative_change_vs_raw":
                rec["d1_ff_relative_change_vs_raw"],
            "ff_ok": bool(ff_ok), "cell_PASS": cell_ok,
        }
    verdict["prediction_PASS"] = bool(all_ok)
    summary["verdict"] = verdict

    summary["caveats"] = [
        "the labeling is a one-time global artifact (per-checkpoint keep-set "
        "frozen as npz); no per-view or per-query logic exists at consumption "
        "— insert §R3 framing guard respected",
        "k and m are absolute view counts calibrated on toy (60 label views) "
        "and applied to courtyard (fewer train views); transfer of absolute "
        "thresholds across view budgets is part of what the courtyard cells "
        "test",
        "depth consistency uses the triangle CENTER only; large triangles "
        "whose center is occluded but whose rim is load-bearing can be "
        "over-removed (kept fractions + ff shifts quantify this)",
        "sub-mesh occupancy is a strict subset of raw route-i occupancy, so "
        "false-occupied can only improve; the risk axis is false-free (the "
        "pre-registered <=10% relative worsening arm)",
        "courtyard GT = laser-scan points: unscanned voxels count GT-free; "
        "GT-collision counts are a lower bound (affects all cells equally)",
        "GTREF floor of 1-2/100 is the planner's own corner-clip floor "
        "(R-03); the collision cap is defined relative to it",
        "D4: labels consume TRAIN views only; toy GT enters only through the "
        "sanctioned calibrate-once objective on toy clean30k",
    ]
    summary["wallclock_sec_total"] = time.time() - t_study
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(_json_safe(summary), f, indent=1)
    print(f"[r3b] wrote {os.path.join(out_root, 'summary.json')} "
          f"({summary['wallclock_sec_total']:.0f}s total)")
    print(f"[r3b] PREDICTION {'PASS' if all_ok else 'FAIL'}")
    return summary


# ---------------------------------------------------------------------------
# Self-test (numpy only; no GPU, no scene assets)
# ---------------------------------------------------------------------------

def selftest():
    # -- keep_mask semantics --
    support = np.array([0, 1, 2, 3, 5, 60], dtype=np.int16)
    cons = np.array([0, 1, 1, 2, 0, 60], dtype=np.int16)
    finite = np.array([True, True, True, True, True, False])
    got = keep_mask(support, cons, finite, k=2, m=1)
    # t0: support 0 <2 no; t1: support 1 <2 no; t2: 2>=2,1>=1 yes;
    # t3: yes; t4: cons 0 <1 no; t5: non-finite no.
    assert got.tolist() == [False, False, True, True, False, False], got
    got = keep_mask(support, cons, finite, k=3, m=2)
    assert got.tolist() == [False, False, False, True, False, False], got
    # cons can never exceed support by construction upstream; keep_mask does
    # not assume it, but check monotonicity: raising k or m never adds faces.
    base = keep_mask(support, cons, finite, 2, 1)
    for k, m in itertools.product((2, 3, 5), (1, 2)):
        sub = keep_mask(support, cons, finite, k, m)
        assert not (sub & ~base).any()

    # -- subset property: sub-mesh occupancy is a subset of full occupancy --
    roi = {"min": [0.0, 0.0], "max": [4.0, 4.0], "z_band": [0.0, 1.0]}
    grid = _VoxelGrid(roi, 0.10)
    verts = np.array([
        [-5, -5, 0.05], [5, -5, 0.05], [5, 5, 0.05], [-5, 5, 0.05],  # ground
        [1.0, 1.0, 0.5], [1.4, 1.0, 0.5], [1.2, 1.4, 0.5],            # floater
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
    finite3 = np.ones(3, dtype=bool)
    occ_full = submesh_occupancy(grid, verts, faces, finite3)
    keep_nofloat = np.array([True, True, False])
    occ_sub = submesh_occupancy(grid, verts, faces, keep_nofloat)
    assert not (occ_sub & ~occ_full).any(), "sub-mesh occupied outside full"
    assert occ_full.sum() > occ_sub.sum()
    # d1 vs a GT that has only the ground: removing the floater must reduce
    # false-occupied and cannot reduce coverage of GT-occupied voxels there.
    gt_occ = submesh_occupancy(grid, verts, faces, keep_nofloat)
    c_full = d1_confusion(gt_occ, occ_full)
    c_sub = d1_confusion(gt_occ, occ_sub)
    assert c_sub["false_occupied_rate"] <= c_full["false_occupied_rate"]
    assert c_sub["false_free_rate"] == 0.0

    # -- calibration selection logic on a synthetic table --
    rows = [
        # (k, m, spurious, coll, ff, fo) — cap coll <= 2.0, fo <= 0.05
        dict(grid_index=0, k=2, m=1, spurious_infeasibility_rate=0.2,
             collisions_per_100_plans=3.0, false_free_rate=0.60,
             false_occupied_rate=0.03),                       # coll infeasible
        dict(grid_index=1, k=2, m=2, spurious_infeasibility_rate=0.3,
             collisions_per_100_plans=1.0, false_free_rate=0.62,
             false_occupied_rate=0.03),                       # feasible
        dict(grid_index=2, k=3, m=1, spurious_infeasibility_rate=0.3,
             collisions_per_100_plans=1.0, false_free_rate=0.61,
             false_occupied_rate=0.03),                       # feasible, lower ff
        dict(grid_index=3, k=5, m=2, spurious_infeasibility_rate=1.0,
             collisions_per_100_plans=None, false_free_rate=0.9,
             false_occupied_rate=0.0),                        # vacuous coll
    ]
    for r in rows:
        r["feasible"] = ((r["collisions_per_100_plans"] is None
                          or r["collisions_per_100_plans"] <= 2.0)
                         and r["false_occupied_rate"] <= 0.05)
    feasible = [r for r in rows if r["feasible"]]

    def sort_key(r):
        coll = r["collisions_per_100_plans"]
        return (r["spurious_infeasibility_rate"],
                coll if coll is not None else float("inf"),
                r["false_free_rate"], r["grid_index"])

    best = min(feasible, key=sort_key)
    assert (best["k"], best["m"]) == (3, 1), best  # ff tie-break decides

    # -- v2 comparison detects mismatches and passes on identity --
    a = [{"problem": 0, "found": True, "reason": "ok", "n_expansions": 5,
          "n_segments": 2, "n_switches": 0, "path_length_m": 1.234567890123}]
    b = [dict(a[0])]
    assert compare_gtref_records(a, b) == []
    b2 = [dict(a[0], n_expansions=6)]
    assert compare_gtref_records(a, b2)
    b3 = [dict(a[0], path_length_m=1.234567890123 + 1e-6)]
    assert compare_gtref_records(a, b3)

    print("certified_submesh selftest PASSED")


def main():
    ap = argparse.ArgumentParser(
        description="GEMS R3.b certified structural sub-mesh")
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
