#!/usr/bin/env python
"""TOPCONF EXP-T2B (GOAL #E-11): second-standard-benchmark collector.

Scenes = the exact 3DGS eval suite (T&T truck/train + DB drjohnson/playroom),
run through the UNCHANGED frozen pipeline (clean30k anchor, PJ-2026 single
transport, final routed stack). Emits per-scene paired CIs + 4-scene
stratified and scene-cluster means. Transfer/external-validity rows: no
gates; reported as measured.
"""
import json
import os
import sys

G1 = "/data/peilincai/gems_stage1"
REPO = "/data/peilincai/mesh-splatting"
SCENES = ["tandt_truck", "tandt_train", "db_drjohnson", "db_playroom"]


def row(name):
    with open(os.path.join(G1, "eval", name, "metrics.json")) as fh:
        return json.load(fh)


def main():
    sys.path.insert(0, REPO)
    import numpy as np
    from tools.gems.paired_bootstrap import summarize_pair
    from tools.ecr.e0_report import stratified_mean_ci, hierarchical_mean_ci

    out = os.path.join(G1, "analysis", "final_stack")
    report = {"per_scene": {}, "suite4": {}}
    md = ["# EXP-T2B — Tanks&Temples + Deep Blending (the 3DGS eval suite)",
          "", "Frozen pipeline transfer: clean30k anchor -> PJ-2026 floor ->"
          " final routed stack; per-scene paired CIs (10k, seed 0);"
          " suite means stratified AND scene-cluster.", "",
          "| scene | anchor PSNR | PJ-2026 PSNR | final PSNR/LPIPS "
          "| dPSNR PJ-vs-anchor [CI] | dPSNR final-vs-PJ [CI] "
          "| dLPIPS final-vs-PJ [CI] |", "|---|---|---|---|---|---|---|"]

    def cmp(a, b, metric, floor):
        pa = a["rendering"]["per_view"]
        pb = b["rendering"]["per_view"]
        assert pa["image_names"] == pb["image_names"]
        return summarize_pair(np.array(pa[metric]), np.array(pb[metric]),
                              floor=floor)

    def fmt(s):
        return f"{s['mean_diff']:+.3f} [{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"

    diffs = {k: {} for k in ("pj_anchor_psnr", "fin_pj_psnr", "fin_pj_lpips",
                             "fin_anchor_psnr")}
    for s in SCENES:
        anc = row(f"{s}_clean30k_v1")
        pj = row(f"e0_{s}_clean30k_pj2026_v1")
        fin = row(f"final_{s}_clean30k_v1")
        sp_pa = cmp(pj, anc, "psnr", 0.10)
        sp_fp = cmp(fin, pj, "psnr", 0.10)
        sl_fp = cmp(fin, pj, "lpips", 0.004)
        for key, (a, b, m) in {
            "pj_anchor_psnr": (pj, anc, "psnr"),
            "fin_pj_psnr": (fin, pj, "psnr"),
            "fin_pj_lpips": (fin, pj, "lpips"),
            "fin_anchor_psnr": (fin, anc, "psnr"),
        }.items():
            pa = np.array(a["rendering"]["per_view"][m])
            pb = np.array(b["rendering"]["per_view"][m])
            diffs[key][s] = (pa - pb).tolist()
        report["per_scene"][s] = {
            "anchor_psnr": anc["rendering"]["mean"]["psnr"],
            "pj_psnr": pj["rendering"]["mean"]["psnr"],
            "final": fin["rendering"]["mean"],
            "pj_vs_anchor_psnr": sp_pa,
            "final_vs_pj_psnr": sp_fp,
            "final_vs_pj_lpips": sl_fp,
            "covered_fraction_mean": float(np.mean(
                [v["covered_fraction"] for v in fin["ecr"]["per_view"]])),
        }
        e = report["per_scene"][s]
        md.append(
            f"| {s} | {e['anchor_psnr']:.3f} | {e['pj_psnr']:.3f} "
            f"| {e['final']['psnr']:.3f}/{e['final']['lpips']:.4f} "
            f"| {fmt(sp_pa)} | {fmt(sp_fp)} | {fmt(sl_fp)} |")

    md += ["", "**Suite-4 means:**", ""]
    for key, label in (("pj_anchor_psnr", "PJ-2026 vs anchor dPSNR"),
                       ("fin_pj_psnr", "final vs PJ-2026 dPSNR"),
                       ("fin_pj_lpips", "final vs PJ-2026 dLPIPS"),
                       ("fin_anchor_psnr", "final vs anchor dPSNR")):
        st = stratified_mean_ci(diffs[key])
        hi = hierarchical_mean_ci(diffs[key])
        report["suite4"][key] = {"stratified": st, "hierarchical": hi}
        md.append(f"- {label}: stratified **{st['mean']:+.4f} "
                  f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]**; scene-cluster "
                  f"{hi['mean']:+.4f} [{hi['ci_lo']:+.4f},{hi['ci_hi']:+.4f}]")
    md += ["", "Coverage: " + ", ".join(
        f"{s} {report['per_scene'][s]['covered_fraction_mean']:.3f}"
        for s in SCENES)]
    with open(os.path.join(out, "t2b_tandt_db.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out, "t2b_tandt_db.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
