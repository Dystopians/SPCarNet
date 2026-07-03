#!/usr/bin/env python
"""GEMS Stage-2 E5-DOWN extension: N=500 maneuvers + ESDF error table
(LEDGER GOAL #015 PART B).

Extends the R3.c planner closed loop (tools/gems/planner_loop.py, LEDGER GOAL
#R-03) from 100 to 500 seed-0 problems per scene, on the SAME R3.a occupancy
grids, with the harness UNCHANGED (all frozen constants imported; no fork).

Cells (N=500 each) per scene {toy_parking, courtyard}:
  GTREF + {clean30k, B50_importance_ft_e1v2_40000} x route-i   (BAR cells)
  + the same models x route-ii                       (SUPPLEMENTARY arms)

Verification before any planning (abort on mismatch):
  v1  GT occupancy rebuilt and bit-exact vs stored R3.a per-voxel indicators
      + stored seed-0 d2 GT verdicts (planner_loop.verify_gt_grid).
  v2  problems[:100] of the N=500 seed-0 sampler == the N=100 seed-0 sampler
      output (same stream; element-wise assert).
  v3  replay pairing: our per-problem records on problems[:100] equal the
      stored R3.c cell_metrics.json records field-by-field (found, reason,
      n_expansions, path_length_m, n_segments, n_switches, gt_collision) on
      every re-planned cell.

ESDF error table (per grid; 2 scenes x 2 models x routes i/ii): footprint
layer (d2 z-band collapse) -> ESDF = distance_transform_edt x 0.10 m (the
exact planner costmap quantity) vs the GT ESDF; |error| mean/median/P90/P95/
max + SIGNED mean over (a) all GT-free cells of the 2D footprint layer and
(b) the near-obstacle band ESDF_GT <= 5 m.

Pre-registered predictions (LEDGER GOAL #015, frozen before any number):
  P-B1 preservation: clean vs B50 route-i outcomes IDENTICAL per problem on
       both scenes (paired found-diff and collision-diff CIs = [0,0]).
  P-B2 conservatism: route-i spurious infeasibility >= 90% toy, = 100%
       courtyard; toy route-i collisions-per-100 stays 0 where evaluable.
  P-B3 supplementary: courtyard route-ii collisions-per-100 >= 5x the
       courtyard GTREF floor at N=500; toy route-ii collision-free.
  P-B4 ESDF (direction-only measurement): signed mean (ESDF_model - ESDF_GT)
       < 0 over GT-free cells on all route-i rows.

Usage:
    python tools/gems/e5_down_ext.py            # full study (CPU only)

Durable: /data/peilincai/gems_stage1/analysis/e5_down_ext/
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.gems.planner_loop import (                    # noqa: E402
    GTREF_LABEL,
    GridMaps,
    PrimitiveTable,
    R3A_ROOT,
    ROUTES,
    SEED,
    VOXEL_M,
    _json_safe,
    footprint_layer,
    load_r3a_cell,
    paired_collision_ci,
    paired_found_ci,
    path_length_inflation,
    plan_cell,
    sample_problems,
    verify_gt_grid,
)
from tools.gems.downstream_metrics import _VoxelGrid, _build_gt_occupancy  # noqa: E402
from tools.gems.occupancy_routes import build_gt_arg     # noqa: E402

N_PROBLEMS_EXT = 500
LABELS = ("clean30k", "B50_importance_ft_e1v2_40000")
SCENES_ORDER = ("toy_parking", "courtyard")
R3C_ROOT = "/data/peilincai/gems_stage1/analysis/r3c_planner"
OUT_ROOT_DEFAULT = "/data/peilincai/gems_stage1/analysis/e5_down_ext"
NEAR_BAND_M = 5.0
REPLAY_FIELDS = ("found", "reason", "n_expansions", "path_length_m",
                 "n_segments", "n_switches", "gt_collision")


def esdf_error_row(esdf_model, esdf_gt, gt_band):
    """|ESDF_model - ESDF_GT| stats over GT-free cells (2D footprint layer),
    full ROI and the near-obstacle band ESDF_GT <= NEAR_BAND_M."""
    free = ~gt_band
    out = {}
    for name, mask in (
            ("gt_free", free),
            (f"gt_free_esdf_le_{NEAR_BAND_M:g}m", free & (esdf_gt <= NEAR_BAND_M))):
        d = esdf_model[mask] - esdf_gt[mask]
        ad = np.abs(d)
        out[name] = {
            "n_cells": int(mask.sum()),
            "abs_mean_m": float(ad.mean()),
            "abs_median_m": float(np.median(ad)),
            "abs_p90_m": float(np.percentile(ad, 90)),
            "abs_p95_m": float(np.percentile(ad, 95)),
            "abs_max_m": float(ad.max()),
            "signed_mean_m": float(d.mean()),
        }
    return out


def load_r3c_records(key):
    path = os.path.join(R3C_ROOT, key, "cell_metrics.json")
    with open(path) as f:
        return json.load(f)["per_problem"], path


def replay_check(key, per_problem_500):
    """v3: field-by-field equality of our first-100 records vs stored R3.c."""
    stored, path = load_r3c_records(key)
    assert len(stored) == 100
    mismatches = []
    for k, (a, b) in enumerate(zip(per_problem_500[:100], stored)):
        for f in REPLAY_FIELDS:
            va = a.get(f)
            vb = b.get(f)
            if isinstance(va, float) and isinstance(vb, float):
                ok = math.isclose(va, vb, rel_tol=0, abs_tol=1e-9)
            else:
                ok = va == vb
            if not ok:
                mismatches.append((k, f, va, vb))
    return {"stored_path": path, "n_compared": 100,
            "n_mismatches": len(mismatches),
            "mismatches": mismatches[:10]}


def run_study(out_root=OUT_ROOT_DEFAULT):
    os.makedirs(out_root, exist_ok=True)
    t_study = time.time()
    print("[e5ext] building primitive table ...", flush=True)
    ptab = PrimitiveTable()

    from tools.gems.scenes import SCENES

    summary = {
        "goal": "LEDGER GOAL #015 PART B (E5-DOWN N=500 extension + ESDF table)",
        "harness": "tools/gems/planner_loop.py UNCHANGED (imported; no fork)",
        "n_problems": N_PROBLEMS_EXT,
        "seed": SEED,
        "verification": {}, "scenes": {}, "cells": {}, "comparisons": {},
        "esdf_table": {}, "verdict": {},
    }

    per_scene_records = {}
    for scene in SCENES_ORDER:
        spec = SCENES[scene]
        grid = _VoxelGrid(spec.roi, VOXEL_M)
        print(f"[e5ext] === scene {scene} ===", flush=True)

        cells_npz = {}
        for label in LABELS:
            arrs, path = load_r3a_cell(scene, label)
            cells_npz[label] = arrs
            print(f"[e5ext]   loaded {path}", flush=True)

        # v1: GT rebuild + bit-exact verification (abort inside on mismatch)
        gt_occ = _build_gt_occupancy(grid, build_gt_arg(spec))
        ok, checks = verify_gt_grid(scene, grid, gt_occ, cells_npz)
        summary["verification"][f"{scene}__v1_gt_bit_exact"] = {
            "all_bit_exact": ok, **checks}
        if not ok:
            raise RuntimeError(f"GT verification FAILED for {scene}: {checks}")
        print(f"[e5ext]   v1 GT bit-exact: {ok} ({len(checks)} checks)", flush=True)

        gt_band = footprint_layer(gt_occ, grid)
        maps_gt = GridMaps(gt_band, ptab)

        # v2: seed-0 stream prefix identity
        problems_100 = sample_problems(maps_gt, np.random.default_rng(SEED),
                                       n_problems=100)
        problems = sample_problems(maps_gt, np.random.default_rng(SEED),
                                   n_problems=N_PROBLEMS_EXT)
        prefix_ok = problems[:100] == problems_100
        summary["verification"][f"{scene}__v2_seed0_prefix_identity"] = bool(prefix_ok)
        if not prefix_ok:
            raise RuntimeError(f"{scene}: N=500 sampler prefix != N=100 sampler")
        print(f"[e5ext]   v2 sampler prefix identity: True "
              f"({len(problems)} problems)", flush=True)

        seps = [math.hypot((p["start"][0] - p["goal"][0]) * VOXEL_M,
                           (p["start"][1] - p["goal"][1]) * VOXEL_M)
                for p in problems]
        summary["scenes"][scene] = {
            "grid_shape": [maps_gt.nx, maps_gt.ny],
            "gt_lethal_fraction": float(maps_gt.lethal.mean()),
            "gt_occupied_fraction": float(gt_band.mean()),
            "n_problems": len(problems),
            "mean_separation_m": float(np.mean(seps)),
        }

        grids_to_plan = [(GTREF_LABEL, None, maps_gt)]
        for label in LABELS:
            for route in ROUTES:
                band = footprint_layer(cells_npz[label][f"occ_route_{route}"],
                                       grid)
                grids_to_plan.append((label, route, GridMaps(band, ptab)))

        per_cell = {}
        for label, route, maps in grids_to_plan:
            key = (f"{scene}__{GTREF_LABEL}" if route is None
                   else f"{scene}__{label}__route_{route}")
            t0 = time.time()
            goal_cells = [p["goal"][0] * maps.ny + p["goal"][1]
                          for p in problems]
            dist_maps = maps.dijkstra_from(goal_cells)
            metrics, per_problem = plan_cell(key, maps, gt_band, problems,
                                             ptab, dist_maps)
            metrics["bar_cell"] = (route == "i") or route is None
            metrics["cell_overhead_sec"] = round(
                time.time() - t0 - sum(r["wall_s"] for r in per_problem), 1)
            # v3: replay pairing vs the stored R3.c first-100 records
            rp = replay_check(key, per_problem)
            summary["verification"][f"{key}__v3_replay_first100"] = _json_safe(rp)
            if rp["n_mismatches"]:
                raise RuntimeError(f"{key}: replay mismatch vs R3.c: "
                                   f"{rp['mismatches'][:3]}")
            per_cell[key] = (metrics, per_problem, maps)
            print(f"[e5ext]   {key}: found {metrics['plans_found']}/"
                  f"{metrics['n_problems']}, coll/100 "
                  f"{metrics['collisions_per_100_plans']}, replay OK", flush=True)

        # inflation vs GTREF + write per-cell json
        ref_key = f"{scene}__{GTREF_LABEL}"
        _, per_ref, _ = per_cell[ref_key]
        for key, (metrics, per_problem, maps) in per_cell.items():
            if key != ref_key:
                metrics["path_length_inflation_vs_gtref"] = \
                    path_length_inflation(per_problem, per_ref)
            cdir = os.path.join(out_root, key)
            os.makedirs(cdir, exist_ok=True)
            clean_pp = [{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in per_problem]
            with open(os.path.join(cdir, "cell_metrics.json"), "w") as f:
                json.dump(_json_safe({"cell": key, "metrics": metrics,
                                      "per_problem": clean_pp}), f, indent=1)
            summary["cells"][key] = _json_safe(metrics)

        # pre-registered comparisons
        comp = {}
        b50 = LABELS[1]
        for route in ROUTES:
            ka = f"{scene}__{b50}__route_{route}"
            kb = f"{scene}__clean30k__route_{route}"
            comp[f"P-B1_found_B50_minus_clean__route_{route}"] = _json_safe(
                paired_found_ci(per_cell[ka][1], per_cell[kb][1]))
            comp[f"P-B1_coll_B50_minus_clean__route_{route}"] = _json_safe(
                paired_collision_ci(per_cell[ka][1], per_cell[kb][1]))
            # strongest form: per-problem outcome identity
            ident = all(
                a["found"] == b["found"] and a["reason"] == b["reason"] and
                a.get("gt_collision") == b.get("gt_collision")
                for a, b in zip(per_cell[ka][1], per_cell[kb][1]))
            comp[f"P-B1_outcomes_identical__route_{route}"] = bool(ident)
        for label in LABELS:
            comp[f"coll_route_ii_minus_i__{label}"] = _json_safe(
                paired_collision_ci(per_cell[f"{scene}__{label}__route_ii"][1],
                                    per_cell[f"{scene}__{label}__route_i"][1]))
            comp[f"found_route_ii_minus_i__{label}"] = _json_safe(
                paired_found_ci(per_cell[f"{scene}__{label}__route_ii"][1],
                                per_cell[f"{scene}__{label}__route_i"][1]))
        summary["comparisons"][scene] = comp

        # ESDF error table
        esdf_gt = maps_gt.esdf
        for label, route, maps in grids_to_plan:
            if route is None:
                continue
            key = f"{scene}__{label}__route_{route}"
            summary["esdf_table"][key] = esdf_error_row(
                maps.esdf, esdf_gt, gt_band)
        per_scene_records[scene] = per_cell

    # ---- verdicts vs pre-registered predictions ----
    v = {}
    cells = summary["cells"]
    comps = summary["comparisons"]

    def ci_zero(c):
        return ("ci_lo" in c and c["ci_lo"] == 0.0 and c["ci_hi"] == 0.0)

    v["P-B1_preservation"] = {}
    for scene in SCENES_ORDER:
        c_f = comps[scene]["P-B1_found_B50_minus_clean__route_i"]
        c_c = comps[scene]["P-B1_coll_B50_minus_clean__route_i"]
        v["P-B1_preservation"][scene] = {
            "found_ci": [c_f.get("ci_lo"), c_f.get("ci_hi")],
            "coll_ci": ([c_c.get("ci_lo"), c_c.get("ci_hi")]
                        if "ci_lo" in c_c else
                        f"unevaluable (n_common_found={c_c.get('n_common_found')})"),
            "outcomes_identical":
                comps[scene]["P-B1_outcomes_identical__route_i"],
            "pass": bool(
                ci_zero(c_f) and
                (ci_zero(c_c) or "ci_lo" not in c_c) and
                comps[scene]["P-B1_outcomes_identical__route_i"]),
        }
    v["P-B1_PASS"] = all(x["pass"] for x in v["P-B1_preservation"].values())

    toy_i = [cells[f"toy_parking__{l}__route_i"] for l in LABELS]
    cy_i = [cells[f"courtyard__{l}__route_i"] for l in LABELS]
    toy_coll_ok = all((c["collisions_per_100_plans"] in (None, 0.0))
                      for c in toy_i)
    v["P-B2_conservatism"] = {
        "toy_route_i_spurious_infeasibility": [
            c["spurious_infeasibility_rate"] for c in toy_i],
        "courtyard_route_i_spurious_infeasibility": [
            c["spurious_infeasibility_rate"] for c in cy_i],
        "toy_route_i_coll_per_100": [
            c["collisions_per_100_plans"] for c in toy_i],
        "pass": bool(
            all(c["spurious_infeasibility_rate"] >= 0.90 for c in toy_i) and
            all(c["spurious_infeasibility_rate"] == 1.0 for c in cy_i) and
            toy_coll_ok),
    }
    v["P-B2_PASS"] = v["P-B2_conservatism"]["pass"]

    cy_floor = cells["courtyard__GTREF"]["collisions_per_100_plans"]
    cy_ii = [cells[f"courtyard__{l}__route_ii"] for l in LABELS]
    toy_ii = [cells[f"toy_parking__{l}__route_ii"] for l in LABELS]
    v["P-B3_supplementary"] = {
        "courtyard_gtref_floor_per_100": cy_floor,
        "courtyard_route_ii_coll_per_100": [
            c["collisions_per_100_plans"] for c in cy_ii],
        "toy_route_ii_coll_per_100": [
            c["collisions_per_100_plans"] for c in toy_ii],
        "pass": bool(
            all(c["collisions_per_100_plans"] is not None and
                (cy_floor == 0 or c["collisions_per_100_plans"] >= 5 * cy_floor)
                for c in cy_ii) and
            all(c["collisions_per_100_plans"] in (None, 0.0) for c in toy_ii)),
    }
    v["P-B3_PASS"] = v["P-B3_supplementary"]["pass"]

    signed = {k: r["gt_free"]["signed_mean_m"]
              for k, r in summary["esdf_table"].items() if "route_i" in k}
    v["P-B4_esdf_signed_mean_route_i"] = signed
    v["P-B4_PASS"] = all(x < 0 for x in signed.values())
    summary["verdict"] = v

    # markdown ESDF table
    lines = ["# E5-DOWN ESDF error table (N/A to plans; grid-level)",
             "",
             "| cell | region | n cells | abs mean | abs median | abs P90 | "
             "abs P95 | abs max | signed mean |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, row in summary["esdf_table"].items():
        for region, s in row.items():
            lines.append(
                f"| {key} | {region} | {s['n_cells']} | "
                f"{s['abs_mean_m']:.3f} | {s['abs_median_m']:.3f} | "
                f"{s['abs_p90_m']:.3f} | {s['abs_p95_m']:.3f} | "
                f"{s['abs_max_m']:.3f} | {s['signed_mean_m']:+.3f} |")
    with open(os.path.join(out_root, "esdf_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    summary["wallclock_sec_total"] = time.time() - t_study
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(_json_safe(summary), f, indent=1)
    print(f"[e5ext] wrote {os.path.join(out_root, 'summary.json')} "
          f"({summary['wallclock_sec_total']:.0f}s total)", flush=True)
    print(json.dumps(_json_safe(v), indent=1))
    return summary


if __name__ == "__main__":
    run_study()
