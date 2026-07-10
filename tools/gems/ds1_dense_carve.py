#!/usr/bin/env python
"""GEMS Stage-4 §6 DS-1: dense-carve retry of R3-FINAL (ONE variant, hard
kill; does NOT gate the mainline; verdict updates CONSUMPTION_IMPOSSIBILITY
either way).

R3-FINAL V1 closed as IMPOSSIBILITY x4 with the audit diagnosis: P1 was MET
(FREE-set false-free <= 9.75%) — the 0/100 failure was UNKNOWN starvation
(stride-16 sampling) plus UNKNOWN-as-obstacle semantics, not map unsafety.
The §6 sign-off reopens exactly ONE mechanism, all three knobs frozen here:

  1. ray stride 16 -> 2 (denser carving; same carve weights/thresholds);
  2. FREE-region dilation by r_inf (EDT(~free) <= r_inf, never overwriting
     OCCUPIED) — compensates the planner's own r_inf lethal inflation that
     previously consumed the thin FREE shell;
  3. planner semantics UNKNOWN = traversable at HIGH COST, never free:
     only OCCUPIED blocks; primitives ENDING in an UNKNOWN band cell pay
     UNKNOWN_COST_MULT x their cost (planner_loop.astar cell_cost_mult hook,
     default-off; >=1 so the heuristic stays admissible). UNKNOWN never
     joins the FREE set anywhere.

NO recalibration: the V1-selected frozen params are reused verbatim
(theta_free=-0.5, theta_occ=1.0, v_min=1, r_inf=1.0). GTREF setup, the 100
seeded problems, collision accounting (GT sweep) and the PASS rule
(courtyard >= 30/100 found AND <= 3.0 collisions/100) are byte-identical to
V1. Cells: toy_parking + courtyard x {clean30k, B50}.

Usage: python tools/gems/ds1_dense_carve.py [--gpu N] [--out-root DIR]
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RAY_STRIDE = 2
UNKNOWN_COST_MULT = 5.0  # frozen: strictly dominates the x2 reverse penalty
OUT_ROOT_DEFAULT = "/data/peilincai/gems_stage1/analysis/ds1_dense_carve"
CELLS = [("toy_parking", "clean30k"), ("toy_parking", "B50"),
         ("courtyard", "clean30k"), ("courtyard", "B50")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    from scipy.ndimage import distance_transform_edt
    import tools.gems.planner_loop as planner
    from tools.gems.three_state_occupancy import (
        CHECKPOINTS, Params, VOXEL_M, _gt_occ, _planner_setup, _toy_bar,
        _write_json, build_log_odds_map, free_set_confusion,
        state_fractions, states_from_params)

    params = Params(theta_free=-0.5, theta_occ=1.0, v_min=1, r_inf=1.0)
    out_root = args.out_root
    setup_cache = {}
    cells = []
    for scene_name, model_label in CELLS:
        print(f"[ds1] === {scene_name} / {model_label} ===", flush=True)
        ckpt = CHECKPOINTS[(scene_name, model_label)]
        cdir = os.path.join(out_root, "maps", f"{scene_name}__{model_label}")
        grid, log_odds, evidence, meta = build_log_odds_map(
            ckpt, scene_name, cdir, RAY_STRIDE)
        gt_occ = _gt_occ(scene_name, grid)
        free, occ, unk = states_from_params(log_odds, evidence, params)

        # mechanism knob 2: FREE dilation by r_inf (never overwrite OCC)
        free_d = (distance_transform_edt(~free) * VOXEL_M
                  <= float(params.r_inf) + 1e-9) & ~occ
        unknown_d = ~(free_d | occ)

        conf_raw = free_set_confusion(free, gt_occ)
        conf_dil = free_set_confusion(free_d, gt_occ)
        frac = state_fractions(free_d, occ, unknown_d)
        np.savez_compressed(
            os.path.join(cdir, "states_ds1.npz"), free=free, occupied=occ,
            unknown=unk, free_dilated=free_d, unknown_dilated=unknown_d)

        key = (scene_name, float(params.r_inf))
        setup = setup_cache.get(key)
        if setup is None:
            setup = _planner_setup(scene_name, gt_occ, params.r_inf)
            setup_cache[key] = setup
        grid_s, ptab, gt_band, maps_gt, problems, metrics_ref, per_ref = setup

        # mechanism knob 3: only OCCUPIED blocks; UNKNOWN band pays x5 cost
        band = planner.footprint_layer(occ, grid)
        unk_band = planner.footprint_layer(unknown_d, grid)
        cell_mult = np.where(unk_band.ravel(), UNKNOWN_COST_MULT, 1.0)
        maps = planner.GridMaps(band, ptab)
        goal_cells = [p["goal"][0] * maps.ny + p["goal"][1] for p in problems]
        dist_maps = maps.dijkstra_from(goal_cells)
        pdir = os.path.join(out_root, "planner", f"{scene_name}__{model_label}")
        metrics, per_problem = planner.plan_cell(
            f"{scene_name}__ds1_{model_label}", maps, gt_band, problems,
            ptab, dist_maps, cell_cost_mult=cell_mult)
        metrics["path_length_inflation_vs_gtref"] = \
            planner.path_length_inflation(per_problem, per_ref)
        metrics["gtref"] = {
            "plans_found": metrics_ref["plans_found"],
            "collisions_per_100_plans": metrics_ref["collisions_per_100_plans"],
            "mean_path_length_m": metrics_ref["mean_path_length_m"],
        }
        clean_pp = [{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in per_problem]
        _write_json(os.path.join(pdir, "cell_metrics.json"),
                    {"metrics": metrics, "per_problem": clean_pp})

        record = {
            "cell": "DS-1",
            "scene": scene_name,
            "model_label": model_label,
            "checkpoint_path": os.path.abspath(ckpt),
            "mechanism": {
                "ray_stride": RAY_STRIDE,
                "free_dilation_m": float(params.r_inf),
                "unknown_semantics": "traversable, never free, "
                                     f"x{UNKNOWN_COST_MULT} primitive cost "
                                     "at UNKNOWN-band end cells",
                "recalibration": "NONE (V1 frozen params reused)",
            },
            "params": params.as_dict(),
            "map_meta": meta,
            "confusion_raw_free": conf_raw,
            "confusion_dilated_free": conf_dil,
            "state_fractions_dilated": frac,
            "planner_metrics": metrics,
            "toy_bar_pass": (_toy_bar(metrics, metrics_ref)
                             if scene_name == "toy_parking" else None),
            "courtyard_fix_target_pass": (
                (metrics["plans_found"] >= 30 and
                 metrics["collisions_per_100_plans"] is not None and
                 float(metrics["collisions_per_100_plans"]) <= 3.0)
                if scene_name == "courtyard" else None),
        }
        _write_json(os.path.join(out_root, "cells",
                                 f"{scene_name}__{model_label}.json"), record)
        cells.append(record)
        print(f"[ds1] {scene_name}/{model_label}: found "
              f"{metrics['plans_found']}/100, coll/100 "
              f"{metrics['collisions_per_100_plans']}, "
              f"fix_target={record['courtyard_fix_target_pass']} "
              f"toy_bar={record['toy_bar_pass']}", flush=True)

    courtyard_pass = all(c["courtyard_fix_target_pass"] for c in cells
                         if c["scene"] == "courtyard")
    summary = {
        "goal": "DS-1 (Stage-4 prompt §6; ONE variant, hard kill)",
        "verdict_courtyard_fix_target": "PASS" if courtyard_pass else "FAIL",
        "cells": cells,
    }
    _write_json(os.path.join(out_root, "summary.json"), summary)
    print(f"[ds1] VERDICT: courtyard fix-target "
          f"{'PASS' if courtyard_pass else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
