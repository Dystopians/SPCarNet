#!/usr/bin/env python
"""E1 analysis (GOAL #005): builds the E1 verdict table from run_eval outputs.

Reads metrics.json for clean + {importance_ft, random_ft, importance_noft} rows
at B in {50, 25} for garden and toy_parking, runs the PROTOCOL paired bootstrap
on per-view rendering metrics, applies the pre-registered E1 criteria, and
writes analysis JSON + markdown. All numbers trace to metrics.json paths.

E1 PASS iff on BOTH scenes at B=50: (a) importance_ft vs clean mean dPSNR
>= -0.20 dB, and (b) importance_ft vs importance_noft mean dPSNR >= +0.5 dB
with 95% CI excluding 0. B=25 rows are recorded whatever they are.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.gems.paired_bootstrap import paired_bootstrap_ci  # noqa: E402

EVAL_ROOT = "/data/peilincai/gems_stage1/eval"
CLEAN = {
    "garden": f"{EVAL_ROOT}/garden_clean30k_v2/metrics.json",
    "toy_parking": f"{EVAL_ROOT}/toy_parking_clean30k_v1/metrics.json",
}
MODES = ["importance_ft", "random_ft", "importance_noft"]


def load(path):
    with open(path) as f:
        return json.load(f)


def per_view(m):
    pv = m["rendering"]["per_view"]
    return pv["image_names"], {k: np.asarray(pv[k], dtype=np.float64) for k in ("psnr", "ssim", "lpips")}


def geo_scalar(m):
    g = m.get("geometry", {}) or {}
    d = m.get("downstream", {}) or {}
    def gv(fam, key):
        v = g.get(fam)
        return v.get(key) if isinstance(v, dict) and key in v else None
    def dv(fam, key):
        v = d.get(fam)
        return v.get(key) if isinstance(v, dict) and key in v else None
    return {
        "g1": gv("g1", "value"),
        "g2_m": gv("g2", "value"),
        "g3_comps": gv("g3", "floater_component_count"),
        "g3_frac": gv("g3", "floater_triangle_fraction"),
        "g4_chamfer_m": gv("g4", "chamfer_l1_m"),
        "g4_fscore": gv("g4", "fscore_at_tau"),
        "d1_false_free": dv("d1", "false_free_rate"),
        "d1_false_occ": dv("d1", "false_occupied_rate"),
        "d2_agreement": dv("d2", "agreement_rate"),
    }


def compare(a_names, a, b_names, b, metric):
    assert a_names == b_names, f"view mismatch: {a_names[:3]} vs {b_names[:3]}"
    r = paired_bootstrap_ci(a[metric], b[metric])
    return {"mean_diff": r["mean_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            "excludes_zero": bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)}


def main():
    out = {"rows": {}, "comparisons": {}, "e1": {}}
    md = ["# E1 analysis (GOAL #005)", ""]
    md.append("| scene | B | mode | tris | PSNR | SSIM | LPIPS | g1 | g3 comps | chamfer m | d1 ff | d2 agr | FPS |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    data = {}
    for scene in ("garden", "toy_parking"):
        m = load(CLEAN[scene])
        data[(scene, "clean")] = m
        geo = geo_scalar(m)
        mean = m["rendering"]["mean"]
        md.append(f"| {scene} | 100 | clean | {m['cost']['n_triangles']:,} | {mean['psnr']:.4f} | "
                  f"{mean['ssim']:.4f} | {mean['lpips']:.4f} | {geo['g1']} | {geo['g3_comps']} | "
                  f"{geo['g4_chamfer_m']} | {geo['d1_false_free']} | {geo['d2_agreement']} | "
                  f"{m['cost']['render_fps']:.1f} |")
        for B in ("50", "25"):
            for mode in MODES:
                p = f"{EVAL_ROOT}/{scene}_B{B}_{mode}_e1b/metrics.json"
                if not os.path.isfile(p):
                    md.append(f"| {scene} | {B} | {mode} | MISSING ({p}) | | | | | | | | | |")
                    continue
                m = load(p)
                data[(scene, B, mode)] = m
                geo = geo_scalar(m)
                mean = m["rendering"]["mean"]
                md.append(f"| {scene} | {B} | {mode} | {m['cost']['n_triangles']:,} | {mean['psnr']:.4f} | "
                          f"{mean['ssim']:.4f} | {mean['lpips']:.4f} | {geo['g1']} | {geo['g3_comps']} | "
                          f"{geo['g4_chamfer_m']} | {geo['d1_false_free']} | {geo['d2_agreement']} | "
                          f"{m['cost']['render_fps']:.1f} |")
                out["rows"][f"{scene}_B{B}_{mode}"] = {
                    "metrics_json": p, "n_triangles": m["cost"]["n_triangles"],
                    **{k: mean[k] for k in ("psnr", "ssim", "lpips")}, **geo,
                }

    md += ["", "## Paired bootstrap comparisons (PROTOCOL §5; diff = A − B)", "",
           "| scene | B | A vs B | metric | mean diff | 95% CI | CI excl. 0 |", "|---|---|---|---|---|---|---|"]
    e1_checks = {}
    for scene in ("garden", "toy_parking"):
        cn, cv = per_view(data[(scene, "clean")])
        for B in ("50", "25"):
            rows = {}
            for mode in MODES:
                if (scene, B, mode) in data:
                    rows[mode] = per_view(data[(scene, B, mode)])
            pairs = [("importance_ft", "importance_noft"), ("importance_ft", "random_ft")]
            for a_mode, b_mode in pairs:
                if a_mode in rows and b_mode in rows:
                    for met in ("psnr", "lpips"):
                        c = compare(rows[a_mode][0], rows[a_mode][1], rows[b_mode][0], rows[b_mode][1], met)
                        out["comparisons"][f"{scene}_B{B}_{a_mode}_vs_{b_mode}_{met}"] = c
                        md.append(f"| {scene} | {B} | {a_mode} vs {b_mode} | {met} | {c['mean_diff']:+.4f} | "
                                  f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {c['excludes_zero']} |")
            for mode in ("importance_ft", "importance_noft"):
                if mode in rows:
                    for met in ("psnr", "lpips"):
                        c = compare(rows[mode][0], rows[mode][1], cn, cv, met)
                        out["comparisons"][f"{scene}_B{B}_{mode}_vs_clean_{met}"] = c
                        md.append(f"| {scene} | {B} | {mode} vs clean | {met} | {c['mean_diff']:+.4f} | "
                                  f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {c['excludes_zero']} |")
            # D3 compaction floor check on the prune-only row (>=20% reduction at
            # iso-quality: dPSNR >= -0.10 AND dLPIPS <= +0.005 vs clean).
            if "importance_noft" in rows:
                dp = out["comparisons"][f"{scene}_B{B}_importance_noft_vs_clean_psnr"]
                dl = out["comparisons"][f"{scene}_B{B}_importance_noft_vs_clean_lpips"]
                out.setdefault("compaction_floor", {})[f"{scene}_B{B}_importance_noft"] = {
                    "dpsnr": dp["mean_diff"], "dpsnr_ci": [dp["ci_lo"], dp["ci_hi"]],
                    "dlpips": dl["mean_diff"], "dlpips_ci": [dl["ci_lo"], dl["ci_hi"]],
                    "meets_floor": bool(dp["mean_diff"] >= -0.10 and dl["mean_diff"] <= 0.005),
                }
            if B == "50" and "importance_ft" in rows and "importance_noft" in rows:
                d_ft = out["comparisons"][f"{scene}_B50_importance_ft_vs_importance_noft_psnr"]
                d_cl = out["comparisons"][f"{scene}_B50_importance_ft_vs_clean_psnr"]
                e1_checks[scene] = {
                    "vs_clean_dpsnr": d_cl["mean_diff"], "vs_clean_ok": d_cl["mean_diff"] >= -0.20,
                    "vs_noft_dpsnr": d_ft["mean_diff"],
                    "vs_noft_ok": d_ft["mean_diff"] >= 0.5 and d_ft["excludes_zero"],
                }

    both = all(v["vs_clean_ok"] and v["vs_noft_ok"] for v in e1_checks.values()) and len(e1_checks) == 2
    out["e1"] = {"checks": e1_checks, "PASS": bool(both)}
    md += ["", f"## E1 VERDICT: {'PASS' if both else 'FAIL'}", "",
           json.dumps(e1_checks, indent=1)]

    os.makedirs("/data/peilincai/gems_stage1/analysis", exist_ok=True)
    with open("/data/peilincai/gems_stage1/analysis/e1_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    with open("/data/peilincai/gems_stage1/analysis/e1_summary.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
