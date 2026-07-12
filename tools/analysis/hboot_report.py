#!/usr/bin/env python
"""TOPCONF EXP-HBOOT: hierarchical (scene-cluster) bootstrap re-analysis of
every Stage-4 headline aggregate, reported ALONGSIDE the pre-registered
stratified intervals (which remain the primary claim form).

Reads ONLY banked per-view arrays; recomputes:
  H1 final stack vs PJ-2026 (full9)      — dPSNR, dLPIPS
  H2 final stack vs PRIMARY anchor       — dPSNR
  H3 L6 compact vs full-budget anchor    — dPSNR, dLPIPS
  H4 L6 compact vs PJ-2026 on B50 base   — dPSNR, dLPIPS
  H5 AT-E0: PJ-2026 vs PRIMARY anchor    — dPSNR
Acceptance rule (frozen in TOPCONF_EXECUTION_PLAN.md): if any interval that
excluded 0 under the stratified scheme includes 0 under the hierarchical
scheme, the corresponding claim text is amended — no cherry-picking.
"""
import json
import os
import sys

G1 = "/data/peilincai/gems_stage1"
REPO = "/data/peilincai/mesh-splatting"
FULL9 = ["garden", "bicycle", "flowers", "stump", "treehill",
         "room", "counter", "kitchen", "bonsai"]


def row(name):
    with open(os.path.join(G1, "eval", name, "metrics.json")) as fh:
        return json.load(fh)


def diffs(a_name, b_name, metric):
    import numpy as np
    a, b = row(a_name), row(b_name)
    pa, pb = a["rendering"]["per_view"], b["rendering"]["per_view"]
    assert pa["image_names"] == pb["image_names"], (a_name, b_name)
    return (np.array(pa[metric]) - np.array(pb[metric])).tolist()


def main():
    sys.path.insert(0, REPO)
    from tools.ecr.e0_report import stratified_mean_ci, hierarchical_mean_ci

    headlines = {
        "H1_final_vs_pj2026": (
            {s: (f"l4_{s}_cleanfixed30k_routed_v1",
                 f"e0_{s}_cleanfixed30k_pj2026_v1") for s in FULL9},
            ("psnr", "lpips")),
        "H2_final_vs_primary": (
            {s: (f"l4_{s}_cleanfixed30k_routed_v1",
                 f"{s}_cleanfixed30k_v1") for s in FULL9},
            ("psnr",)),
        "H3_l6_vs_primary_anchor": (
            {s: (f"final_{s}_B50_v1",
                 f"{s}_cleanfixed30k_v1") for s in FULL9},
            ("psnr", "lpips")),
        "H4_l6_vs_pj2026_b50": (
            {s: (f"final_{s}_B50_v1",
                 f"e0_{s}_B50_pj2026_v1") for s in FULL9},
            ("psnr", "lpips")),
        "H5_ate0_pj2026_vs_primary": (
            {s: (f"e0_{s}_cleanfixed30k_pj2026_v1",
                 f"{s}_cleanfixed30k_v1") for s in FULL9},
            ("psnr",)),
    }
    report = {}
    md = ["# Hierarchical (scene-cluster) bootstrap — headline re-analysis",
          "", "Primary claim form remains the pre-registered stratified"
          " mean-of-scene-means CI; the 2-stage scene-cluster interval is"
          " reported alongside (both 10k resamples, seed 0, same banked"
          " per-view arrays).", "",
          "| headline | metric | stratified CI | hierarchical CI | "
          "hier. excl. 0 |", "|---|---|---|---|---|"]
    any_flip = False
    for key, (pairs, metrics) in headlines.items():
        report[key] = {}
        for metric in metrics:
            d = {s: diffs(a, b, metric) for s, (a, b) in pairs.items()}
            st = stratified_mean_ci(d)
            hi = hierarchical_mean_ci(d)
            excl = (hi["ci_lo"] > 0) if st["mean"] > 0 else (hi["ci_hi"] < 0)
            strat_excl = (st["ci_lo"] > 0) if st["mean"] > 0 \
                else (st["ci_hi"] < 0)
            if strat_excl and not excl:
                any_flip = True
            report[key][metric] = {"stratified": st, "hierarchical": hi,
                                   "hierarchical_excludes_zero": bool(excl)}
            fmt = (lambda c: f"{c['mean']:+.4f} "
                   f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}]")
            md.append(f"| {key} | {metric} | {fmt(st)} | {fmt(hi)} "
                      f"| {'YES' if excl else '**NO — claim amended**'} |")
    if any_flip:
        verdict = ("AT LEAST ONE interval no longer excludes 0 — the "
                   "corresponding claim text MUST be amended "
                   "(see acceptance rule).")
    else:
        verdict = ("ALL headline intervals also exclude 0 under "
                   "scene-cluster resampling — conclusions are robust to "
                   "treating the scene as the sampling unit.")
    md += ["", f"**Verdict:** {verdict}"]
    out = os.path.join(G1, "analysis", "final_stack")
    with open(os.path.join(out, "hierarchical_cis.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out, "hierarchical_cis.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
