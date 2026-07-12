#!/usr/bin/env python
"""TOPCONF EXP-IBR (GOAL #E-12): score the IBRNet outputs — external
learned-IBR baseline cell (sanctioned single-mouth exception, E-09/Difix
precedent: mirror identical to run_eval conventions, self-validated by
re-scoring the base PNGs against the banked base row <= 0.01 dB).

IBRNet: pretrained generalizable model (model_255000.pth, no per-scene
training or fine-tuning — stated in the row), 10 source views per target by
the frozen transport camera score (a superset of ECR's evidence rights).
Convention verified by the banked self-reconstruction gate (22.48 dB).
"""
import json
import os
import sys

G1 = "/data/peilincai/gems_stage1"
REPO = "/data/peilincai/mesh-splatting"
SCENES = ["garden", "bicycle", "kitchen"]


def load_row(name):
    with open(os.path.join(G1, "eval", name, "metrics.json")) as fh:
        return json.load(fh)


def main():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
    import numpy as np
    import torch
    import torchvision
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips
    from tools.gems.paired_bootstrap import summarize_pair

    device = torch.device("cuda")

    def img(path):
        t = torchvision.io.read_image(path).float().div(255.0)
        return t[:3].to(device).contiguous()

    out_root = os.path.join(G1, "analysis", "ibr_cell")
    os.makedirs(out_root, exist_ok=True)
    report = {}
    md = ["# GOAL #E-12 — IBRNet (pretrained generalizable, 10 source views)"
          " on the ECR test splits", "",
          "> Context-only external cell (sanctioned mirror exception; mirror"
          " self-validated vs the banked base rows). IBRNet receives MORE"
          " evidence than the ECR transport (10 nearest train GT views vs"
          " calibrated K=2–8) but no per-scene training — the honest"
          " generalizable-IBR point. Convention gate: self-reconstruction"
          " 22.48 dB (banked).", "",
          "| scene | point | PSNR | SSIM | LPIPS | dPSNR vs base [CI] |",
          "|---|---|---|---|---|---|"]
    for scene in SCENES:
        quals = os.path.join(G1, "analysis", "quals", f"{scene}_final")
        jobs_dir = os.path.join(G1, "ibr_cell", f"testset_{scene}")
        fin = load_row(f"l4_{scene}_cleanfixed30k_routed_v1")
        base_row = load_row(f"{scene}_cleanfixed30k_v1")
        names = fin["rendering"]["per_view"]["image_names"]

        rows = {"base": [], "ibr": []}
        with torch.no_grad():
            for n in names:
                gt = img(os.path.join(quals, n, "gt.png"))
                for key, path in (("base", os.path.join(quals, n, "base.png")),
                                  ("ibr", os.path.join(jobs_dir, f"{n}.png"))):
                    x = img(path)
                    rows[key].append({
                        "psnr": psnr(x.unsqueeze(0), gt.unsqueeze(0))
                        .mean().item(),
                        "ssim": ssim(x.unsqueeze(0), gt.unsqueeze(0)).item(),
                        "lpips": lpips(x.unsqueeze(0), gt.unsqueeze(0),
                                       net_type="vgg").item()})
        base_banked = base_row["rendering"]["per_view"]["psnr"]
        max_dev = max(abs(a - b["psnr"])
                      for a, b in zip(base_banked, rows["base"]))
        assert max_dev <= 0.01, f"mirror validation failed: {max_dev}"

        def arr(key, m):
            return np.array([r[m] for r in rows[key]])
        sp = summarize_pair(arr("ibr", "psnr"), arr("base", "psnr"),
                            floor=0.10)
        fm = fin["rendering"]["mean"]
        entry = {
            "mirror_base_max_dev_db": max_dev,
            "ibr": {m: float(arr("ibr", m).mean())
                    for m in ("psnr", "ssim", "lpips")},
            "base": {m: float(arr("base", m).mean())
                     for m in ("psnr", "ssim", "lpips")},
            "ecr_final": {"psnr": fm["psnr"], "ssim": fm["ssim"],
                          "lpips": fm["lpips"]},
            "ibr_vs_base_psnr": sp,
            "per_view": {"names": names,
                         "ibr_psnr": [r["psnr"] for r in rows["ibr"]],
                         "ibr_lpips": [r["lpips"] for r in rows["ibr"]]},
            "config": "IBRNet model_255000 pretrained, 10 sources by frozen "
                      "camera score, coarse+fine 64+64, no fine-tuning",
        }
        report[scene] = entry
        b, i, e = entry["base"], entry["ibr"], entry["ecr_final"]
        md.append(f"| {scene} | base (PRIMARY anchor) | {b['psnr']:.3f} "
                  f"| {b['ssim']:.4f} | {b['lpips']:.4f} | — |")
        md.append(f"| {scene} | IBRNet (generalizable) | {i['psnr']:.3f} "
                  f"| {i['ssim']:.4f} | {i['lpips']:.4f} "
                  f"| {sp['mean_diff']:+.3f} "
                  f"[{sp['ci_lo']:+.3f},{sp['ci_hi']:+.3f}] |")
        md.append(f"| {scene} | ECR final stack (mouth row) | {e['psnr']:.3f} "
                  f"| {e['ssim']:.4f} | {e['lpips']:.4f} | — |")
        with open(os.path.join(out_root, f"ibr_{scene}.json"), "w") as fh:
            json.dump(entry, fh, indent=1)
    md += ["", "_Generated by tools/analysis/ibr_cell.py from the rendered"
           " outputs in gems_stage1/ibr_cell/testset_<scene>/ (adapter:"
           " /data/peilincai/IBRNet/ibr_infer.py; per-frame timings in the"
           " ibr_infer job log)._"]
    with open(os.path.join(out_root, "ibr_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
