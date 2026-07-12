#!/usr/bin/env python
"""Route-A EXP-ORACLE: score every method's edit-region output against TRUE
edited ground truth (the oracle re-render of the synthetic scene without the
deleted element), and validate the real-scene proxy metric (leak_R) by
correlation with oracle error.

Inputs: the --save-outputs dump of edit_eval.py (all methods × all test
views + regions.npz) and the oracle dataset's images/ directory (verified by
tools/edit/verify_oracle_build.py BEFORE use — camera byte-identity etc.).
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True,
                    help="edit_eval --save-outputs dir")
    ap.add_argument("--oracle-images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import numpy as np
    import torch
    import torchvision
    from scipy.stats import spearmanr
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from tools.gems.paired_bootstrap import summarize_pair

    regions = np.load(os.path.join(args.outputs, "regions.npz"))
    names = sorted({k.split("__")[0] for k in regions.files})
    methods = sorted(d for d in os.listdir(args.outputs)
                     if os.path.isdir(os.path.join(args.outputs, d)))

    def img(path):
        return torchvision.io.read_image(path).float().div(255.0)[:3] \
            .contiguous().cuda()

    def masked_psnr(a, b, m):
        d2 = ((a - b) ** 2).mean(0)
        mse = float((d2 * m).sum() / m.sum().clamp(min=1))
        return 10.0 * np.log10(1.0 / max(mse, 1e-10))

    def masked_mae(a, b, m):
        d = (a - b).abs().mean(0)
        return float((d * m).sum() / m.sum().clamp(min=1))

    per_method = {m: {"oracle_psnr_R": [], "oracle_mae_R": [],
                      "oracle_psnr_U": []} for m in methods}
    with torch.no_grad():
        for n in names:
            oracle = img(os.path.join(args.oracle_images, f"{n}.png"))
            R8 = torch.from_numpy(regions[f"{n}__R8"]).cuda()
            U = torch.from_numpy(regions[f"{n}__U"]).cuda()
            for m in methods:
                x = img(os.path.join(args.outputs, m, f"{n}.png"))
                per_method[m]["oracle_psnr_R"].append(
                    masked_psnr(x, oracle, R8))
                per_method[m]["oracle_mae_R"].append(masked_mae(x, oracle, R8))
                per_method[m]["oracle_psnr_U"].append(
                    masked_psnr(x, oracle, U))

    # proxy validation: leak_R (deviation from edited base) vs oracle MAE,
    # rank correlation over (method, view) pairs excluding the references
    c1 = "C1_editedbase"
    proxy_pairs, oracle_pairs = [], []
    with torch.no_grad():
        for n in names:
            base = img(os.path.join(args.outputs, c1, f"{n}.png"))
            R8 = torch.from_numpy(regions[f"{n}__R8"]).cuda()
            for m in methods:
                if m in (c1, "ORIG_ecr"):
                    continue
                x = img(os.path.join(args.outputs, m, f"{n}.png"))
                proxy_pairs.append(masked_mae(x, base, R8))
                oracle_pairs.append(
                    per_method[m]["oracle_mae_R"][names.index(n)])
    rho, pval = spearmanr(proxy_pairs, oracle_pairs)

    def ci(a, b):
        return summarize_pair(np.array(a), np.array(b), floor=0.10)

    report = {"methods": {m: {k: float(np.mean(v)) for k, v in d.items()}
                          for m, d in per_method.items()},
              "proxy_validation": {
                  "spearman_rho_leak_vs_oracle_mae": float(rho),
                  "p_value": float(pval),
                  "n_pairs": len(proxy_pairs)},
              "cis_oracle_psnr_R": {}}
    if "C5_ours" in methods:
        for m in methods:
            if m in ("C5_ours", "ORIG_ecr"):
                continue
            report["cis_oracle_psnr_R"][f"C5_minus_{m}"] = ci(
                per_method["C5_ours"]["oracle_psnr_R"],
                per_method[m]["oracle_psnr_R"])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "oracle_eval.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    md = ["# EXP-ORACLE — true-edited-GT scoring", "",
          "| method | oracle PSNR_R ↑ | oracle MAE_R ↓ | oracle PSNR_U ↑ |",
          "|---|---|---|---|"]
    for m in methods:
        d = report["methods"][m]
        md.append(f"| {m} | {d['oracle_psnr_R']:.3f} "
                  f"| {d['oracle_mae_R']:.4f} | {d['oracle_psnr_U']:.3f} |")
    md += ["", f"**Proxy validation:** Spearman ρ(leak_R, oracle MAE_R) = "
           f"**{rho:.3f}** (p = {pval:.2e}, n = {len(proxy_pairs)})", "",
           "**Paired CIs, oracle PSNR_R (C5 − alternative):**"]
    for k, v in report["cis_oracle_psnr_R"].items():
        md.append(f"- {k}: {v['mean_diff']:+.3f} "
                  f"[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]")
    with open(os.path.join(args.out, "oracle_eval.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
