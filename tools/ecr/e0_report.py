#!/usr/bin/env python
"""GEMS Stage-4 M-E0 report: PJ-2026 floor rows vs the three references.

Reads e0_<scene>_<tag>_pj2026_v1 rows + banked base rows, computes per-scene
paired CIs (tools/gems/paired_bootstrap, the ONLY CI implementation) and the
full9-mean stratified CI for the AT-E0 gate (resample views WITHIN each scene,
mean of scene means; numpy seed 0, 10k resamples — same discipline, mean-of-
scene-means unit because "full9 mean" weights scenes equally). Writes
markdown + json under analysis/e0_pj2026/.
"""
import argparse
import json
import os
import sys

REPO = "/data/peilincai/mesh-splatting"
G1 = "/data/peilincai/gems_stage1"
FULL9 = ["garden", "bicycle", "flowers", "stump", "treehill",
         "room", "counter", "kitchen", "bonsai"]
TOWNS = ["ss3dm_town01", "ss3dm_town02", "ss3dm_town03", "ss3dm_town06"]

# banked reference rows (verified on disk, STATUS audit + asset survey)
PRIMARY_ROW = {s: f"{s}_cleanfixed30k_v1" for s in FULL9}
PRIMARY_ROW.update({s: f"{s}_clean30k_v1" for s in TOWNS + ["toy_parking"]})
LEGACY_ROW = {"garden": "garden_clean30k_v2"}
LEGACY_ROW.update({s: f"{s}_clean30k_v1" for s in FULL9 if s != "garden"})
B50_ROW = {s: f"{s}_B50_importance_ft_s2" for s in FULL9 + TOWNS}
B50_ROW["garden"] = "garden_B50_importance_ft_e1v2"
B50_ROW["toy_parking"] = "toy_parking_B50_importance_ft_e1v2"


def load_row(name):
    path = os.path.join(G1, "eval", name, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def per_view(row, metric):
    return row["rendering"]["per_view"][metric]


def paired(row_a, row_b, metric, floor):
    import numpy as np
    from tools.gems.paired_bootstrap import summarize_pair
    names_a = row_a["rendering"]["per_view"]["image_names"]
    names_b = row_b["rendering"]["per_view"]["image_names"]
    assert names_a == names_b, "view sets differ — not pairable"
    return summarize_pair(np.array(per_view(row_a, metric)),
                          np.array(per_view(row_b, metric)), floor=floor)


def stratified_mean_ci(diffs_by_scene, n_resamples=10000, seed=0):
    """CI of the mean-of-scene-means of paired per-view differences."""
    import numpy as np
    rng = np.random.default_rng(seed)
    scenes = sorted(diffs_by_scene)
    arrays = [np.asarray(diffs_by_scene[s], dtype=np.float64) for s in scenes]
    point = float(np.mean([a.mean() for a in arrays]))
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        means[i] = np.mean([
            a[rng.integers(0, len(a), size=len(a))].mean() for a in arrays])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": point, "ci_lo": float(lo), "ci_hi": float(hi)}


def hierarchical_mean_ci(diffs_by_scene, n_resamples=10000, seed=0):
    """Two-stage (cluster) bootstrap: resample SCENES with replacement, then
    views within each drawn scene. Treats scene as the sampling unit, so the
    interval also carries scene-to-scene variance (wider than the stratified
    interval by construction). Reported ALONGSIDE stratified_mean_ci, never
    replacing it (TOPCONF EXP-HBOOT, 2026-07-11)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    scenes = sorted(diffs_by_scene)
    arrays = [np.asarray(diffs_by_scene[s], dtype=np.float64) for s in scenes]
    n_scenes = len(arrays)
    point = float(np.mean([a.mean() for a in arrays]))
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        picks = rng.integers(0, n_scenes, size=n_scenes)
        means[i] = np.mean([
            arrays[j][rng.integers(0, len(arrays[j]),
                                   size=len(arrays[j]))].mean()
            for j in picks])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "scheme": "scene-cluster + within-scene, 2-stage"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=("primary", "b50"), default="primary")
    ap.add_argument("--out", default=os.path.join(G1, "analysis", "e0_pj2026"))
    args = ap.parse_args()
    sys.path.insert(0, REPO)
    import numpy as np

    if args.base == "primary":
        tag = {s: ("cleanfixed30k" if s in FULL9 else "clean30k")
               for s in FULL9 + TOWNS + ["toy_parking"]}
        base_row_of = PRIMARY_ROW
    else:
        tag = {s: "B50" for s in FULL9 + TOWNS + ["toy_parking"]}
        base_row_of = B50_ROW

    os.makedirs(args.out, exist_ok=True)
    results = {}
    diffs_psnr, diffs_lpips = {}, {}
    for scene in FULL9 + TOWNS + ["toy_parking"]:
        ecr = load_row(f"e0_{scene}_{tag[scene]}_pj2026_v1")
        if ecr is None:
            results[scene] = {"status": "MISSING"}
            continue
        entry = {"status": "OK",
                 "ecr_psnr": ecr["rendering"]["mean"]["psnr"],
                 "ecr_ssim": ecr["rendering"]["mean"]["ssim"],
                 "ecr_lpips": ecr["rendering"]["mean"]["lpips"],
                 "alpha": ecr["ecr"]["alpha"],
                 "cost": {k: ecr["cost"].get(k) for k in (
                     "cache_mb_raw", "cache_mb_compressed",
                     "transport_ms_per_frame", "end_to_end_fps",
                     "render_fps", "total_artifact_mb", "disk_mb")}}
        refs = {"vs_base": base_row_of[scene]}
        if args.base == "primary":
            refs["vs_primary"] = PRIMARY_ROW[scene]
            if scene in LEGACY_ROW:
                refs["vs_legacy"] = LEGACY_ROW[scene]
        else:
            refs["vs_primary"] = PRIMARY_ROW[scene]
        for key, row_name in refs.items():
            ref = load_row(row_name)
            if ref is None:
                entry[key] = {"status": f"reference row missing: {row_name}"}
                continue
            entry[key] = {
                "reference_row": row_name,
                "ref_psnr": ref["rendering"]["mean"]["psnr"],
                "psnr": paired(ecr, ref, "psnr", 0.10),
                "lpips": paired(ecr, ref, "lpips", 0.004),
            }
            if key == "vs_primary" and scene in FULL9:
                a = np.array(per_view(ecr, "psnr"))
                b = np.array(per_view(ref, "psnr"))
                diffs_psnr[scene] = (a - b).tolist()
                al = np.array(per_view(ecr, "lpips"))
                bl = np.array(per_view(ref, "lpips"))
                diffs_lpips[scene] = (al - bl).tolist()
        results[scene] = entry

    summary = {"base": args.base, "per_scene": results}
    if len(diffs_psnr) == 9:
        summary["full9_mean_dpsnr_vs_primary"] = stratified_mean_ci(diffs_psnr)
        summary["full9_mean_dlpips_vs_primary"] = stratified_mean_ci(diffs_lpips)
        at = summary["full9_mean_dpsnr_vs_primary"]
        summary["AT_E0"] = {
            "bar": ">= +1.0 dB over PRIMARY anchor on full9 mean, CI excl. 0",
            "value": at,
            "pass": bool(at["ci_lo"] > 0.0 and at["mean"] >= 1.0),
        }
    out_json = os.path.join(args.out, f"e0_{args.base}_summary.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1)

    lines = [f"# M-E0 PJ-2026 floor — base = {args.base}", ""]
    lines.append("| scene | ECR PSNR | dPSNR vs primary [CI] | dLPIPS vs primary [CI] | dPSNR vs base [CI] | alpha | transport ms | e2e fps | cache MB raw/comp |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for scene in FULL9 + TOWNS + ["toy_parking"]:
        e = results[scene]
        if e.get("status") != "OK":
            lines.append(f"| {scene} | {e.get('status')} | | | | | | | |")
            continue
        vp = e.get("vs_primary", {})
        vb = e.get("vs_base", {})

        def fmt(block, metric):
            if "psnr" not in block:
                return "n/a"
            s = block[metric]
            return f"{s['mean_diff']:+.3f} [{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"
        c = e["cost"]
        lines.append(
            f"| {scene} | {e['ecr_psnr']:.4f} | {fmt(vp,'psnr')} | {fmt(vp,'lpips')} "
            f"| {fmt(vb,'psnr')} | {e['alpha']} | {c['transport_ms_per_frame']:.0f} "
            f"| {c['end_to_end_fps']:.2f} | {c['cache_mb_raw']:.0f}/{c['cache_mb_compressed']:.0f} |")
    if "AT_E0" in summary:
        at = summary["AT_E0"]
        lines += ["", f"**full9 mean dPSNR vs primary = {at['value']['mean']:+.4f} "
                      f"[{at['value']['ci_lo']:+.4f},{at['value']['ci_hi']:+.4f}] "
                      f"→ AT-E0 {'PASS' if at['pass'] else 'FAIL'}** "
                      f"(bar: {at['bar']})"]
        dl = summary["full9_mean_dlpips_vs_primary"]
        lines.append(f"full9 mean dLPIPS vs primary = {dl['mean']:+.5f} "
                     f"[{dl['ci_lo']:+.5f},{dl['ci_hi']:+.5f}]")
    out_md = os.path.join(args.out, f"e0_{args.base}_table.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
