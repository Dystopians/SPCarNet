#!/usr/bin/env python
"""Pre-submission master table for Route A (edit-aware ECR) — the canonical,
oracle-primary presentation replacing every leak_R-first table.

Statistical annotations (task-3 requirements, printed in the table header):
- pairing unit = TEST VIEW; all CIs are paired per-view bootstrap
  (10,000 resamples, seed 0, percentile method);
- effect directions defined per metric;
- multiple comparisons: the 5-way novelty family (C5 vs each alternative)
  is additionally reported at 99% CIs (Bonferroni-style alpha = 0.05/5);
- sample sizes per scene stated; affected-view counts reconciled.

Oracle scope: TRUE edited GT exists ONLY for the synthetic scene (verified
oracle rebuild). Real scenes report content preservation (true-GT outside
the edit region) and BOUNDED ghost metrics only — no edited-GT access is
implied. leak_R is a SECONDARY bounded-deviation check (rho = 0.502 vs
oracle error; it penalizes legitimate improvement).
"""
import json
import os
import sys

REPO = "/data/peilincai/mesh-splatting"
G1 = "/data/peilincai/gems_stage1"


def main():
    sys.path.insert(0, REPO)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
    import numpy as np
    import torch
    import torchvision

    outdir = os.path.join(G1, "analysis", "edit_aware")
    outputs = os.path.join(outdir, "abl_toy_outputs")
    regions = np.load(os.path.join(outputs, "regions.npz"))
    names = sorted({k.split("__")[0] for k in regions.files})
    methods = sorted(d for d in os.listdir(outputs)
                     if os.path.isdir(os.path.join(outputs, d)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def img(p):
        return torchvision.io.read_image(p).float().div(255.0)[:3] \
            .contiguous().to(device)

    def mpsnr(a, b, m):
        d2 = ((a - b) ** 2).mean(0)
        mse = float((d2 * m).sum() / m.sum().clamp(min=1))
        return 10.0 * np.log10(1.0 / max(mse, 1e-10))

    per = {m: [] for m in methods}
    with torch.no_grad():
        for n in names:
            oracle = img(os.path.join(
                G1, "datasets", "toy_parking_nocar0", "images", f"{n}.png"))
            R8 = torch.from_numpy(regions[f"{n}__R8"]).to(device)
            for m in methods:
                per[m].append(mpsnr(img(os.path.join(outputs, m, f"{n}.png")),
                                    oracle, R8))

    rng_seed = 0

    def paired_ci(a, b, levels=((2.5, 97.5), (0.5, 99.5))):
        rng = np.random.default_rng(rng_seed)
        d = np.array(a) - np.array(b)
        n = len(d)
        means = np.array([d[rng.integers(0, n, n)].mean()
                          for _ in range(10000)])
        out = {"mean": float(d.mean()), "n": n}
        for lo, hi in levels:
            l, h = np.percentile(means, [lo, hi])
            out[f"ci{int(100 - 2 * lo)}"] = [float(l), float(h)]
        return out

    novelty = {}
    alts = ["ABL_dilate4", "ABL_dilate16", "ABL_box2d", "TM_targetmask",
            "C4_rebuild"]
    for alt in alts:
        novelty[alt] = paired_ci(per["C5_ours"], per[alt])

    # real-scene cells (bounded metrics only) from banked evals
    def load(cell):
        return json.load(open(os.path.join(outdir, cell, "edit_eval.json")))

    cells = {
        "garden table delete (2,037,550 faces)": load("abl_garden"),
        "garden table recolor": load(os.path.join("..", "edit_aware",
                                                  "garden_recolor"))
        if False else json.load(open(os.path.join(
            outdir, "garden_recolor", "edit_eval.json"))),
        "garden pot delete (peripheral, 24,952 faces)": load("garden_delpot"),
        "garden chained delete->recolor": load("garden_chain2"),
        "toy car_1 delete (711,609+ faces)": load("toy_delcar1"),
    }

    md = ["# Route-A MASTER TABLE (pre-submission canonical form)", "",
          "**Statistics:** pairing unit = test view; paired per-view "
          "bootstrap, 10,000 resamples, seed 0, percentile CIs. Effect "
          "directions: oracle PSNR_R higher = better edited-region fidelity "
          "to TRUE edited GT; ghost_psnr_R higher = MORE stale-content "
          "similarity (worse); psnr_U higher = better true-GT preservation "
          "outside the region. **Multiple comparisons:** the 5-way novelty "
          "family is reported at BOTH 95% and 99% CIs (Bonferroni alpha = "
          "0.05/5 = 0.01). **Oracle scope:** true edited GT exists ONLY for "
          "the synthetic scene (verified rebuild); real-scene cells report "
          "content preservation + bounded ghost metrics — no edited-GT "
          "access is implied. leak_R is SECONDARY (rho = 0.502 vs oracle "
          "error; penalizes legitimate improvement).", "",
          "## A. Oracle-scored novelty family (toy car_0 deletion; "
          f"n = {len(names)} test views)", "",
          "| C5 (ours) minus | Δ oracle PSNR_R | 95% CI | 99% CI "
          "| excl. 0 @95 / @99 |", "|---|---|---|---|---|"]
    for alt, s in novelty.items():
        e95 = s["ci95"][0] > 0 or s["ci95"][1] < 0
        e99 = s["ci99"][0] > 0 or s["ci99"][1] < 0
        md.append(f"| {alt} | {s['mean']:+.3f} "
                  f"| [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] "
                  f"| [{s['ci99'][0]:+.3f},{s['ci99'][1]:+.3f}] "
                  f"| {'Y' if e95 else 'N'} / {'Y' if e99 else 'N'} |")
    s_stale = paired_ci(per["C5_ours"], per["C2_stale"])
    md += ["", f"Honest non-member of the family: C5 − C2_stale = "
           f"{s_stale['mean']:+.3f} [{s_stale['ci95'][0]:+.3f},"
           f"{s_stale['ci95'][1]:+.3f}] (tie on DELETION; C2 fails recolor "
           "+1.964 [+1.869,+2.061] and chained recolor +2.669 "
           "[+1.326,+4.085] — bounded ghost metric, real scene).", ""]

    md += ["## B. Real-scene cells (bounded metrics; no edited GT)", "",
           "| cell | n views | C5 leak_R (secondary) | ghost C5−C1 [95% CI] "
           "| U preservation C5−ORIG [95% CI] (true GT) |",
           "|---|---|---|---|---|"]
    for label, d in cells.items():
        n = d["n_views"]
        c = d["cis"]
        g = c["ghost_C5_minus_C1"]
        p = c["presU_C5_minus_ORIG"]
        md.append(
            f"| {label} | {n} | {d['per_method']['C5_ours']['leak_R']:.4f} "
            f"| {g['mean_diff']:+.3f} [{g['ci_lo']:+.3f},{g['ci_hi']:+.3f}] "
            f"| {p['mean_diff']:+.3f} [{p['ci_lo']:+.3f},{p['ci_hi']:+.3f}] |")

    md += ["", "## C. Update cost & affected-view reconciliation", "",
           "| cell | affected / train views | note | bytes (dense) "
           "| bytes (sparse sidecar) | wall |", "|---|---|---|---|---|---|"]
    rec = [
        ("garden table (central)", "161 / 161", "central object: visible in "
         "every train view", "1053 MB", "n/a (dense run)", "108 s"),
        ("garden pot (peripheral)", "57 / 161", "TRUE view-locality: 35% of "
         "views", "369 MB", "n/a (dense run)", "42 s"),
        ("toy car_0", "72 / 72", "72 = ALL of toy's TRAIN views (its 90 "
         "total views include 18 test; the dataset census's '76/90' counts "
         "coverage over ALL views incl. test)", "231 MB",
         "**12.8 MB (validated bit-equal, same process)**", "34 s"),
    ]
    for r in rec:
        md.append("| " + " | ".join(r) + " |")
    md += ["", "Reconciliation: 57/161 and 72/72 are DIFFERENT SCENES and "
           "denominators — garden has 161 train views (pot affects 57); "
           "toy_parking has 72 train views (car_0 affects all 72; the "
           "widely-quoted 76/90 figure is the dataset's whole-set coverage "
           "census including test views)."]

    with open(os.path.join(outdir, "routeA_master_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    json.dump({"novelty_oracle": novelty,
               "stale_tie": s_stale,
               "n_views_oracle": len(names)},
              open(os.path.join(outdir, "routeA_master_table.json"), "w"),
              indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
