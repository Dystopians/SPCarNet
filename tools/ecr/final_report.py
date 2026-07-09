#!/usr/bin/env python
"""GEMS Stage-4 Tier-1 report generator (prompt §5).

Emits, from banked rows only (zero hand-typed numbers):
  T-ECR-1  final-stack quality vs THREE references (legacy clean@30k,
           PRIMARY clean-fixed@30k, PJ-2026 floor) — per-scene paired CIs,
           win/loss, full9 stratified means (+ suites when banked)
  T-ECR-2  efficiency/cost — render_fps, transport ms/frame, end-to-end fps,
           cache raw/compressed MB, checkpoint MB, total artifact MB
  T-ECR-3  ladder/ablation — per-rung deltas from the banked gate jsons
Writes markdown+json under analysis/final_stack/.
"""
import argparse
import json
import os
import sys

REPO = "/data/peilincai/mesh-splatting"
G1 = "/data/peilincai/gems_stage1"
FULL9 = ["garden", "bicycle", "flowers", "stump", "treehill",
         "room", "counter", "kitchen", "bonsai"]
SUITES = ["ss3dm_town01", "ss3dm_town02", "ss3dm_town03", "ss3dm_town06",
          "toy_parking"]

FINAL_ROW = {s: f"l4_{s}_cleanfixed30k_routed_v1" for s in FULL9}
FINAL_ROW.update({s: f"final_{s}_clean30k_v1" for s in SUITES})
L6_ROW = {s: f"final_{s}_B50_v1" for s in FULL9}
PJ_ROW = {s: f"e0_{s}_cleanfixed30k_pj2026_v1" for s in FULL9}
PJ_ROW.update({s: f"e0_{s}_clean30k_pj2026_v1" for s in SUITES})
PRIMARY_ROW = {s: f"{s}_cleanfixed30k_v1" for s in FULL9}
PRIMARY_ROW.update({s: f"{s}_clean30k_v1" for s in SUITES})
LEGACY_ROW = {"garden": "garden_clean30k_v2"}
LEGACY_ROW.update({s: f"{s}_clean30k_v1" for s in FULL9 if s != "garden"})
LEGACY_ROW.update({s: f"{s}_clean30k_v1" for s in SUITES})


def load_row(name):
    path = os.path.join(G1, "eval", name, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(G1, "analysis", "final_stack"))
    args = ap.parse_args()
    sys.path.insert(0, REPO)
    import numpy as np
    from tools.gems.paired_bootstrap import summarize_pair
    from tools.ecr.e0_report import stratified_mean_ci

    os.makedirs(args.out, exist_ok=True)
    report = {"per_scene": {}, "full9": {}, "l6": {}}
    md = ["# GEMS Stage-4 — FINAL ECR STACK, Tier-1 tables",
          "", "All CIs: paired per-view bootstrap (10k, seed 0); full9 means:"
          " stratified mean-of-scene-means bootstrap. References named per"
          " column (prompt §0).", "",
          "## T-ECR-1 quality (final stack = v3 routed)", "",
          "| scene | final PSNR/SSIM/LPIPS | dPSNR vs legacy [CI] | dPSNR vs"
          " primary [CI] | dPSNR vs PJ-2026 [CI] | dLPIPS vs PJ-2026 [CI] |",
          "|---|---|---|---|---|---|"]

    def cmp(row_a, row_b, metric, floor):
        pa, pb = row_a["rendering"]["per_view"], row_b["rendering"]["per_view"]
        assert pa["image_names"] == pb["image_names"]
        return summarize_pair(np.array(pa[metric]), np.array(pb[metric]),
                              floor=floor)

    def fmt(s):
        return f"{s['mean_diff']:+.3f} [{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"

    diffs = {"psnr": {"legacy": {}, "primary": {}, "pj": {}},
             "lpips": {"legacy": {}, "primary": {}, "pj": {}}}
    wins = {"legacy": 0, "primary": 0, "pj": 0}
    for scene in FULL9 + SUITES:
        fin = load_row(FINAL_ROW[scene])
        if fin is None:
            md.append(f"| {scene} | PENDING | | | | |")
            continue
        entry = {"psnr": fin["rendering"]["mean"]["psnr"],
                 "ssim": fin["rendering"]["mean"]["ssim"],
                 "lpips": fin["rendering"]["mean"]["lpips"]}
        cells = {}
        for key, rows in (("legacy", LEGACY_ROW), ("primary", PRIMARY_ROW),
                          ("pj", PJ_ROW)):
            ref = load_row(rows[scene])
            if ref is None:
                cells[key] = {"psnr": None, "lpips": None}
                continue
            sp = cmp(fin, ref, "psnr", 0.10)
            sl = cmp(fin, ref, "lpips", 0.004)
            cells[key] = {"psnr": sp, "lpips": sl, "ref_row": rows[scene]}
            if scene in FULL9:
                a = np.array(fin["rendering"]["per_view"]["psnr"])
                b = np.array(load_row(rows[scene])["rendering"]["per_view"]["psnr"])
                diffs["psnr"][key][scene] = (a - b).tolist()
                al = np.array(fin["rendering"]["per_view"]["lpips"])
                bl = np.array(load_row(rows[scene])["rendering"]["per_view"]["lpips"])
                diffs["lpips"][key][scene] = (al - bl).tolist()
                if sp["ci_lo"] > 0:
                    wins[key] += 1
        entry["vs"] = {k: {m: (v[m] if not isinstance(v[m], dict) else v[m])
                           for m in ("psnr", "lpips")}
                       for k, v in cells.items()}
        report["per_scene"][scene] = entry
        md.append(
            f"| {scene} | {entry['psnr']:.3f}/{entry['ssim']:.4f}/"
            f"{entry['lpips']:.4f} "
            f"| {fmt(cells['legacy']['psnr']) if cells['legacy']['psnr'] else 'n/a'} "
            f"| {fmt(cells['primary']['psnr']) if cells['primary']['psnr'] else 'n/a'} "
            f"| {fmt(cells['pj']['psnr']) if cells['pj']['psnr'] else 'n/a'} "
            f"| {fmt(cells['pj']['lpips']) if cells['pj']['lpips'] else 'n/a'} |")

    if all(len(diffs["psnr"][k]) == 9 for k in ("legacy", "primary", "pj")):
        md += ["", "**full9 means (stratified):**", ""]
        for key, label in (("legacy", "legacy clean@30k"),
                           ("primary", "PRIMARY clean-fixed@30k"),
                           ("pj", "PJ-2026 floor")):
            mp = stratified_mean_ci(diffs["psnr"][key])
            ml = stratified_mean_ci(diffs["lpips"][key])
            report["full9"][key] = {"dpsnr": mp, "dlpips": ml,
                                    "psnr_ci_wins": wins[key]}
            md.append(f"- vs {label}: dPSNR **{mp['mean']:+.4f} "
                      f"[{mp['ci_lo']:+.4f},{mp['ci_hi']:+.4f}]**, dLPIPS "
                      f"**{ml['mean']:+.5f} [{ml['ci_lo']:+.5f},"
                      f"{ml['ci_hi']:+.5f}]**, per-scene PSNR CI-wins "
                      f"{wins[key]}/9")

    # ---- T-ECR-2 efficiency ----
    md += ["", "## T-ECR-2 efficiency / honest cost (final stack rows)", "",
           "| scene (base) | render fps | transport ms | e2e fps | cache MB "
           "raw/comp | ckpt MB | TOTAL artifact MB |", "|---|---|---|---|---|---|---|"]
    for scene in FULL9 + SUITES:
        for label, rows in (("primary", FINAL_ROW), ("B50", L6_ROW)):
            if scene in SUITES and label == "B50":
                continue
            row = load_row(rows.get(scene, ""))
            if row is None:
                continue
            c = row["cost"]
            md.append(
                f"| {scene} ({label}) | {c['render_fps']:.1f} "
                f"| {c.get('transport_ms_per_frame', -1):.0f} "
                f"| {c.get('end_to_end_fps', -1):.2f} "
                f"| {c.get('cache_mb_raw', -1):.0f}/"
                f"{c.get('cache_mb_compressed', -1):.0f} "
                f"| {c['disk_mb']:.0f} "
                f"| {c.get('total_artifact_mb', -1):.0f} |")
    md += ["", "Timing caveat: shared GPUs (contention, cf. Stage-2 T4 note)."]

    # ---- T-ECR-3 ladder (= the per-rung ablation table) ----
    md += ["", "## T-ECR-3 ladder / ablation (per-rung deltas, full9 means)",
           "", "| rung | vs | dPSNR [CI] | dLPIPS [CI] | verdict |",
           "|---|---|---|---|---|"]
    for name, path in (("L1b distilled base", "l1_gate.json"),
                       ("L2 multiband K-source", "l2_gate.json"),
                       ("L3 learned fusion", "l3_gate.json"),
                       ("L4 routing", "l4_gate.json"),
                       ("FINAL vs PJ-2026", "l4_vs_floor.json")):
        p = os.path.join(G1, "analysis", "e0_pj2026", path)
        if not os.path.exists(p):
            continue
        g = json.load(open(p))
        mp, ml = g["full9_mean_dpsnr"], g["full9_mean_dlpips"]
        md.append(f"| {name} | {g['incumbent'].split('{')[0]}… | "
                  f"{mp['mean']:+.4f} [{mp['ci_lo']:+.4f},{mp['ci_hi']:+.4f}] | "
                  f"{ml['mean']:+.5f} [{ml['ci_lo']:+.5f},{ml['ci_hi']:+.5f}] | "
                  f"{g['verdict']} |")

    # ---- L6 compact tie-back ----
    l6_diffs_p, l6_diffs_l = {}, {}
    for scene in FULL9:
        fin = load_row(L6_ROW[scene])
        pj = load_row(PRIMARY_ROW[scene])
        if fin is None or pj is None:
            continue
        a = np.array(fin["rendering"]["per_view"]["psnr"])
        b = np.array(pj["rendering"]["per_view"]["psnr"])
        l6_diffs_p[scene] = (a - b).tolist()
        al = np.array(fin["rendering"]["per_view"]["lpips"])
        bl = np.array(pj["rendering"]["per_view"]["lpips"])
        l6_diffs_l[scene] = (al - bl).tolist()
    if len(l6_diffs_p) == 9:
        mp = stratified_mean_ci(l6_diffs_p)
        ml = stratified_mean_ci(l6_diffs_l)
        report["l6"] = {"dpsnr_vs_primary_anchor": mp,
                        "dlpips_vs_primary_anchor": ml}
        md += ["", "## L6 compact tie-back (final stack on B5@B50, HALF the "
               "triangles, vs the FULL-BUDGET primary anchor)", "",
               f"full9 mean dPSNR **{mp['mean']:+.4f} [{mp['ci_lo']:+.4f},"
               f"{mp['ci_hi']:+.4f}]**, dLPIPS **{ml['mean']:+.5f} "
               f"[{ml['ci_lo']:+.5f},{ml['ci_hi']:+.5f}]**"]

    out_md = os.path.join(args.out, "final_stack_tables.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(args.out, "final_stack_summary.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n".join(md))
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
