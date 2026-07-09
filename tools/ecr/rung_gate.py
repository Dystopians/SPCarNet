#!/usr/bin/env python
"""GEMS Stage-4 ladder rung gate (PROTOCOL 1.2.0; prompt §1 raised floors).

Compares a candidate stack's rows against the incumbent stack's rows on
full9 and prints the promotion verdict:

    PROMOTE iff full9 mean dPSNR >= +0.10 dB OR dLPIPS improves >= 0.004
    vs the incumbent, CI excl. 0 (stratified per-view bootstrap,
    mean-of-scene-means, seed 0, 10k resamples).

Usage:
    python -m tools.ecr.rung_gate \
        --candidate 'l1_{scene}_distill_pj2026_v1' \
        --incumbent 'e0_{scene}_cleanfixed30k_pj2026_v1' \
        --label L1b --out analysis/e0_pj2026/l1_gate.json
"""
import argparse
import json
import os
import sys

REPO = "/data/peilincai/mesh-splatting"
G1 = "/data/peilincai/gems_stage1"
FULL9 = ["garden", "bicycle", "flowers", "stump", "treehill",
         "room", "counter", "kitchen", "bonsai"]


def load_row(name):
    path = os.path.join(G1, "eval", name, "metrics.json")
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="row-name pattern with {scene}")
    ap.add_argument("--incumbent", required=True,
                    help="row-name pattern with {scene}")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.path.insert(0, REPO)
    import numpy as np
    from tools.gems.paired_bootstrap import summarize_pair
    from tools.ecr.e0_report import stratified_mean_ci

    per_scene = {}
    d_psnr, d_lpips = {}, {}
    for scene in FULL9:
        cand = load_row(args.candidate.format(scene=scene))
        inc = load_row(args.incumbent.format(scene=scene))
        pc, pi = cand["rendering"]["per_view"], inc["rendering"]["per_view"]
        assert pc["image_names"] == pi["image_names"], f"{scene}: unpaired"
        a_p, b_p = np.array(pc["psnr"]), np.array(pi["psnr"])
        a_l, b_l = np.array(pc["lpips"]), np.array(pi["lpips"])
        per_scene[scene] = {
            "cand_psnr": cand["rendering"]["mean"]["psnr"],
            "inc_psnr": inc["rendering"]["mean"]["psnr"],
            "psnr": summarize_pair(a_p, b_p, floor=0.10),
            "lpips": summarize_pair(a_l, b_l, floor=0.004),
        }
        d_psnr[scene] = (a_p - b_p).tolist()
        d_lpips[scene] = (a_l - b_l).tolist()

    m_psnr = stratified_mean_ci(d_psnr)
    m_lpips = stratified_mean_ci(d_lpips)
    psnr_pass = m_psnr["ci_lo"] > 0.0 and m_psnr["mean"] >= 0.10
    lpips_pass = m_lpips["ci_hi"] < 0.0 and (-m_lpips["mean"]) >= 0.004
    verdict = "PROMOTE" if (psnr_pass or lpips_pass) else "BELOW FLOOR (DIAGNOSTIC)"

    report = {
        "label": args.label,
        "candidate": args.candidate,
        "incumbent": args.incumbent,
        "full9_mean_dpsnr": m_psnr,
        "full9_mean_dlpips": m_lpips,
        "psnr_floor_pass": bool(psnr_pass),
        "lpips_floor_pass": bool(lpips_pass),
        "verdict": verdict,
        "per_scene": per_scene,
    }
    print(f"=== RUNG GATE {args.label} ===")
    for scene in FULL9:
        s = per_scene[scene]
        print(f"  {scene:9s} dPSNR {s['psnr']['mean_diff']:+.3f} "
              f"[{s['psnr']['ci_lo']:+.3f},{s['psnr']['ci_hi']:+.3f}]  "
              f"dLPIPS {s['lpips']['mean_diff']:+.4f} "
              f"[{s['lpips']['ci_lo']:+.4f},{s['lpips']['ci_hi']:+.4f}]")
    print(f"full9 mean dPSNR {m_psnr['mean']:+.4f} "
          f"[{m_psnr['ci_lo']:+.4f},{m_psnr['ci_hi']:+.4f}] "
          f"(floor +0.10, pass={psnr_pass})")
    print(f"full9 mean dLPIPS {m_lpips['mean']:+.5f} "
          f"[{m_lpips['ci_lo']:+.5f},{m_lpips['ci_hi']:+.5f}] "
          f"(floor 0.004 improve, pass={lpips_pass})")
    print(f"VERDICT: {verdict}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
