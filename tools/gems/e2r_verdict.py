"""GEMS Stage-1R R2 — E2R-v1 verdict computation vs the B5 model.

Consumes the single-mouth eval rows (metrics.json + per-sample npz) of the
E2R model and its B5 baseline; computes the pre-registered PASS-bar numbers
(GOAL #R-00 / insert R2):

  - ΔPSNR ≥ −0.10 dB vs B5 (paired per-view bootstrap CI, PROTOCOL §5);
  - (g1 OR d1-false-free OR g3) improves ≥30% relative vs B5, CI excl. 0
    (g1: paired per-evidence-sample violations; d1-ff: paired per-GT-occupied
    -voxel indicators; g3: component counts, no natural pairing — relative
    change reported WITHOUT CI, as in GOAL #008);
  - removed-triangle counts + opacity distributions before/after (from the
    checkpoints) for the LEDGER entry.

No metric fork: every number is arithmetic over run_eval.py outputs plus
tools.gems.paired_bootstrap (the single CI implementation).

Usage:
    python -m tools.gems.e2r_verdict \
        --scene toy_parking \
        --b5-eval  <eval dir of B5 row> --e2r-eval <eval dir of E2R row> \
        --b5-ckpt  <B5 point_cloud_state_dict.pt> \
        --e2r-ckpt <E2R point_cloud_state_dict.pt> \
        --out <analysis dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.gems.paired_bootstrap import summarize_pair  # noqa: E402


def _load_metrics(eval_dir):
    with open(os.path.join(eval_dir, "metrics.json")) as f:
        return json.load(f)


def _rel(delta, base):
    return float(delta) / base if base not in (0, 0.0) else float("nan")


def _opacity_stats(ckpt_path):
    import torch
    s = torch.load(ckpt_path, map_location="cpu")
    floor = float(s.get("opacity_floor", 0.999))
    w = s["vertex_weight"].float()
    real = floor + (1.0 - floor) * torch.sigmoid(w)
    q = torch.quantile(real.reshape(-1), torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    return {
        "opacity_floor": floor,
        "n_triangles": int(s["_triangle_indices"].shape[0]),
        "n_vertices": int(s["triangles_points"].shape[0]),
        "realized_min": float(real.min()),
        "realized_mean": float(real.mean()),
        "realized_quantiles_1_5_25_50_75_95_99": [float(x) for x in q],
        "frac_below_0.05": float((real < 0.05).float().mean()),
        "frac_below_0.5": float((real < 0.5).float().mean()),
        "frac_above_0.9": float((real > 0.9).float().mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--b5-eval", required=True)
    p.add_argument("--e2r-eval", required=True)
    p.add_argument("--b5-ckpt", required=True)
    p.add_argument("--e2r-ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    b5 = _load_metrics(args.b5_eval)
    e2 = _load_metrics(args.e2r_eval)
    out = {"scene": args.scene, "b5_eval": os.path.abspath(args.b5_eval),
           "e2r_eval": os.path.abspath(args.e2r_eval)}

    # ---------------- rendering: paired per-view PSNR / LPIPS ----------------
    names_b5 = b5["rendering"]["per_view"]["image_names"]
    names_e2 = e2["rendering"]["per_view"]["image_names"]
    assert names_b5 == names_e2, "test-view sets differ; pairing impossible"
    psnr = summarize_pair(np.array(e2["rendering"]["per_view"]["psnr"]),
                          np.array(b5["rendering"]["per_view"]["psnr"]))
    lpips = summarize_pair(np.array(e2["rendering"]["per_view"]["lpips"]),
                           np.array(b5["rendering"]["per_view"]["lpips"]))
    out["rendering"] = {
        "n_views": len(names_b5),
        "psnr_b5": b5["rendering"]["mean"]["psnr"],
        "psnr_e2r": e2["rendering"]["mean"]["psnr"],
        "dpsnr_e2r_minus_b5": psnr,
        "lpips_b5": b5["rendering"]["mean"]["lpips"],
        "lpips_e2r": e2["rendering"]["mean"]["lpips"],
        "dlpips_e2r_minus_b5": lpips,
        "psnr_guard_pass": bool(psnr["mean_diff"] >= -0.10),
        "psnr_guard_ci_lo_above_-0.10": bool(psnr["ci_lo"] >= -0.10),
    }

    # ---------------- g1: paired per-sample free-space violations ------------
    g1 = {"present": "g1" in b5.get("geometry", {}) and "g1" in e2.get("geometry", {})}
    if g1["present"]:
        za = np.load(os.path.join(args.e2r_eval, "geometry/g1_free_space_samples.npz"))
        zb = np.load(os.path.join(args.b5_eval, "geometry/g1_free_space_samples.npz"))
        aligned = (za["cam_index"].shape == zb["cam_index"].shape
                   and bool(np.array_equal(za["cam_index"], zb["cam_index"]))
                   and bool(np.array_equal(za["px"], zb["px"]))
                   and bool(np.array_equal(za["py"], zb["py"])))
        g1["samples_aligned"] = aligned
        g1["g1_b5"] = b5["geometry"]["g1"]["value"]
        g1["g1_e2r"] = e2["geometry"]["g1"]["value"]
        if aligned:
            s = summarize_pair(za["violation"].astype(np.float64),
                               zb["violation"].astype(np.float64))
            g1["paired_diff_e2r_minus_b5"] = s
            g1["relative_change"] = _rel(s["mean_diff"], g1["g1_b5"])
            g1["improves_30pct_ci_excl0"] = bool(
                s["mean_diff"] < 0 and s["excludes_zero"]
                and (-g1["relative_change"]) >= 0.30)
    out["g1"] = g1

    # ---------------- d1 false-free: paired per-GT-occupied voxel ------------
    d1 = {"present": "d1" in b5.get("downstream", {}) and "d1" in e2.get("downstream", {})}
    if d1["present"]:
        za = np.load(os.path.join(args.e2r_eval, "downstream/d1_per_sample.npz"))
        zb = np.load(os.path.join(args.b5_eval, "downstream/d1_per_sample.npz"))
        a = za["recon_free_at_gt_occupied"].astype(np.float64)
        b = zb["recon_free_at_gt_occupied"].astype(np.float64)
        d1["n_gt_occupied"] = int(a.shape[0])
        d1["aligned"] = a.shape == b.shape  # same frozen GT grid => same order
        d1["ff_b5"] = b5["downstream"]["d1"]["false_free_rate"]
        d1["ff_e2r"] = e2["downstream"]["d1"]["false_free_rate"]
        d1["fo_b5"] = b5["downstream"]["d1"]["false_occupied_rate"]
        d1["fo_e2r"] = e2["downstream"]["d1"]["false_occupied_rate"]
        if d1["aligned"]:
            s = summarize_pair(a, b)
            d1["paired_ff_diff_e2r_minus_b5"] = s
            d1["relative_change"] = _rel(s["mean_diff"], d1["ff_b5"])
            d1["improves_30pct_ci_excl0"] = bool(
                s["mean_diff"] < 0 and s["excludes_zero"]
                and (-d1["relative_change"]) >= 0.30)
    out["d1_false_free"] = d1

    # ---------------- g3: floater components (no pairing across topologies) --
    g3 = {"present": "g3" in b5.get("geometry", {}) and "g3" in e2.get("geometry", {})}
    if g3["present"]:
        cb, ce = b5["geometry"]["g3"], e2["geometry"]["g3"]
        g3["floater_components_b5"] = cb["floater_component_count"]
        g3["floater_components_e2r"] = ce["floater_component_count"]
        g3["floater_fraction_b5"] = cb["floater_triangle_fraction"]
        g3["floater_fraction_e2r"] = ce["floater_triangle_fraction"]
        g3["relative_change_components"] = _rel(
            ce["floater_component_count"] - cb["floater_component_count"],
            cb["floater_component_count"])
        g3["relative_change_fraction"] = _rel(
            ce["floater_triangle_fraction"] - cb["floater_triangle_fraction"],
            cb["floater_triangle_fraction"])
        g3["improves_30pct"] = bool(g3["relative_change_components"] <= -0.30
                                    or g3["relative_change_fraction"] <= -0.30)
        g3["note"] = ("component/topology identity changes under removal; no "
                      "paired CI (same treatment as GOAL #008 E2v3)")
    out["g3"] = g3

    # ---------------- counts + opacity distributions -------------------------
    out["checkpoints"] = {
        "b5": _opacity_stats(args.b5_ckpt),
        "e2r": _opacity_stats(args.e2r_ckpt),
    }
    out["removed_triangles"] = (out["checkpoints"]["b5"]["n_triangles"]
                                - out["checkpoints"]["e2r"]["n_triangles"])
    out["removal_fraction_of_b5"] = _rel(out["removed_triangles"],
                                         out["checkpoints"]["b5"]["n_triangles"])
    out["budget_respected"] = bool(out["removed_triangles"] >= 0)

    # ---------------- other metrics carried for the report -------------------
    out["carry"] = {}
    for fam, key in (("geometry", "g2"), ("geometry", "g4"), ("downstream", "d2")):
        if key in b5.get(fam, {}) and key in e2.get(fam, {}):
            out["carry"][key] = {"b5": b5[fam][key], "e2r": e2[fam][key]}
    out["cost"] = {"b5": b5.get("cost"), "e2r": e2.get("cost")}

    # ---------------- verdict per pre-registered bar -------------------------
    gbar = bool(
        (g1.get("improves_30pct_ci_excl0") is True)
        or (d1.get("improves_30pct_ci_excl0") is True)
        or (g3.get("improves_30pct") is True))
    out["pass_bar"] = {
        "geometry_or_downstream_30pct": gbar,
        "psnr_guard_mean": out["rendering"]["psnr_guard_pass"],
        "panels": "HUMAN/VISUAL — see eval panels dirs",
        "scene_pass_pending_panels": bool(gbar and out["rendering"]["psnr_guard_pass"]),
    }

    path = os.path.join(args.out, f"e2r_verdict_{args.scene}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["pass_bar"], indent=1))
    print(f"[e2r_verdict] -> {path}")


if __name__ == "__main__":
    main()
