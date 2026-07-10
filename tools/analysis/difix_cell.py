#!/usr/bin/env python
"""GEMS Stage-4 GOAL #E-09: external generative-enhancer cell — Difix3D+
(nvidia/difix_ref, single-step) applied to the PRIMARY base renders.

SANCTIONED single-mouth exception (R1/GOAL#017 precedent): the enhancer is an
external method, so its outputs are scored OUTSIDE run_eval.py by this mirror
— same 8-bit-PNG round-trip convention (inputs here ARE the metric-path PNGs
dumped by tools/analysis/ecr_dump_quals.py), same psnr/ssim/lpips-vgg
implementations imported from this repo, same llff8 split (name-asserted
against the banked rows). The mirror VALIDATES itself by re-scoring the base
PNGs and asserting equality with the banked base row (<=0.01 dB).

Reference rule (frozen): per test view, the reference image = the evidence
cache's train GT of the transport's TOP-scored support view, read from the
banked final-stack row's per-view support_names[0] — the same train-view
evidence rights the ECR transport has (D4-legal input class). Context row:
gates nothing, no cross-method claim; reported per prompt §4b.
"""
import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
G1 = "/data/peilincai/gems_stage1"
SCENES = ["garden", "bicycle", "kitchen"]
GPU = "7"


def load_row(name):
    with open(os.path.join(G1, "eval", name, "metrics.json")) as fh:
        return json.load(fh)


def main():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU)
    import numpy as np
    import torch
    import torchvision
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips
    from tools.gems.paired_bootstrap import summarize_pair

    out_root = os.path.join(G1, "analysis", "difix_cell")
    report = {}
    md = ["# GOAL #E-09 — Difix3D+ (nvidia/difix_ref, single-step) on the "
          "PRIMARY base renders", "",
          "> Context-only external cell (sanctioned single-mouth exception, "
          "R1 precedent — mirror validated against the banked base rows). "
          "Reference per view = train GT of the transport's top support view "
          "(same evidence rights as ECR). CIs: paired per-view bootstrap "
          "(10k, seed 0) vs the base row.", "",
          "| scene | point | PSNR | SSIM | LPIPS | dPSNR vs base [CI] | "
          "dLPIPS vs base [CI] |", "|---|---|---|---|---|---|---|"]

    def img(path, device):
        t = torchvision.io.read_image(path).float().div(255.0)
        return t[:3].to(device)

    device = torch.device("cuda")
    for scene in SCENES:
        quals = os.path.join(G1, "analysis", "quals", f"{scene}_final")
        fin = load_row(f"l4_{scene}_cleanfixed30k_routed_v1")
        base_row = load_row(f"{scene}_cleanfixed30k_v1")
        names = fin["rendering"]["per_view"]["image_names"]
        assert names == base_row["rendering"]["per_view"]["image_names"]
        support = {v["image_name"]: v["support_names"][0]
                   for v in fin["ecr"]["per_view"]}
        cache_gt = os.path.join(
            G1, "ecr_cache", f"{scene}_cleanfixed30k_l4routed", "gt")

        stage = os.path.join(out_root, scene)
        din, dref, dout = (os.path.join(stage, x) for x in
                           ("in", "ref", "out"))
        for d in (din, dref, dout):
            os.makedirs(d, exist_ok=True)
        for n in names:
            shutil.copyfile(os.path.join(quals, n, "base.png"),
                            os.path.join(din, f"{n}.png"))
            shutil.copyfile(os.path.join(cache_gt, f"{support[n]}.png"),
                            os.path.join(dref, f"{n}.png"))

        print(f"[difix_cell] {scene}: inference on {len(names)} views")
        proc = subprocess.run(
            ["bash", os.path.join(G1, "difix", "run_inference.sh"),
             din, dout, dref, GPU],
            capture_output=True, text=True)
        sys.stdout.write(proc.stdout[-2000:])
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-4000:])
            raise SystemExit(f"difix inference failed on {scene}")
        med_line = [l for l in proc.stdout.splitlines() if "median" in l]

        # mirror metrics (+ base validation vs the banked mouth row)
        rows = {"base": [], "difix": []}
        with torch.no_grad():
            for n in names:
                gt = img(os.path.join(quals, n, "gt.png"), device)
                for key, d in (("base", din), ("difix", dout)):
                    x = img(os.path.join(d, f"{n}.png"), device)
                    rows[key].append({
                        "psnr": psnr(x.unsqueeze(0), gt.unsqueeze(0))
                        .mean().item(),
                        "ssim": ssim(x.unsqueeze(0), gt.unsqueeze(0)).item(),
                        "lpips": lpips(x.unsqueeze(0), gt.unsqueeze(0),
                                       net_type="vgg").item()})
        base_banked = base_row["rendering"]["per_view"]["psnr"]
        base_mirror = [r["psnr"] for r in rows["base"]]
        max_dev = max(abs(a - b) for a, b in zip(base_banked, base_mirror))
        print(f"[difix_cell] {scene}: mirror-vs-mouth base max dev "
              f"{max_dev:.5f} dB")
        assert max_dev <= 0.01, "mirror does not reproduce the banked base row"

        def arr(key, m):
            return np.array([r[m] for r in rows[key]])

        sp = summarize_pair(arr("difix", "psnr"), arr("base", "psnr"),
                            floor=0.10)
        sl = summarize_pair(arr("difix", "lpips"), arr("base", "lpips"),
                            floor=0.004)
        fm = fin["rendering"]["mean"]
        entry = {
            "n_views": len(names),
            "mirror_base_max_dev_db": max_dev,
            "difix": {m: float(arr("difix", m).mean())
                      for m in ("psnr", "ssim", "lpips")},
            "base": {m: float(arr("base", m).mean())
                     for m in ("psnr", "ssim", "lpips")},
            "ecr_final": {"psnr": fm["psnr"], "ssim": fm["ssim"],
                          "lpips": fm["lpips"]},
            "difix_vs_base": {"psnr": sp, "lpips": sl},
            "per_view": {"names": names,
                         "difix_psnr": [r["psnr"] for r in rows["difix"]],
                         "difix_lpips": [r["lpips"] for r in rows["difix"]]},
            "timing": med_line[-1] if med_line else None,
            "reference_rule": "top-1 transport support view train GT",
        }
        report[scene] = entry
        d, b, e = entry["difix"], entry["base"], entry["ecr_final"]
        md.append(f"| {scene} | base (PRIMARY anchor) | {b['psnr']:.3f} "
                  f"| {b['ssim']:.4f} | {b['lpips']:.4f} | — | — |")
        md.append(
            f"| {scene} | + Difix3D+ single-step | {d['psnr']:.3f} "
            f"| {d['ssim']:.4f} | {d['lpips']:.4f} "
            f"| {sp['mean_diff']:+.3f} [{sp['ci_lo']:+.3f},{sp['ci_hi']:+.3f}] "
            f"| {sl['mean_diff']:+.4f} [{sl['ci_lo']:+.4f},{sl['ci_hi']:+.4f}] |")
        md.append(f"| {scene} | ECR final stack (v3, mouth row) | "
                  f"{e['psnr']:.3f} | {e['ssim']:.4f} | {e['lpips']:.4f} "
                  f"| — | — |")
        with open(os.path.join(out_root, f"difix_{scene}.json"), "w") as fh:
            json.dump(entry, fh, indent=1)

    md += ["", "Timing: single-step, GPU 7 (shared-GPU caveat). Difix "
           "outputs at input resolution (internal /8 resize + LANCZOS back).",
           "", "_Generated by tools/analysis/difix_cell.py; inference via "
           "gems_stage1/difix/run_inference.sh (DifixPipeline route)._"]
    with open(os.path.join(out_root, "difix_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
