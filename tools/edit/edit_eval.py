#!/usr/bin/env python
"""Route A (edit-aware ECR): the frozen evaluation harness
(EDIT_AWARE_ECR_PROTOCOL.md v1). Outside-the-mouth measurement cell —
edited scenes have no GT — EXCEPT the unaffected-region metrics, which are
computed against REAL test GT (valid there).

Methods (same edited checkpoint):
  C1 edited-base   : base renderer, no ECR
  C2 stale-ECR     : ORIGINAL unmodified cache (fingerprint guard bypassed
                     inside this harness only — the documented failure mode)
  C4 full-rebuild  : cache rebuilt from the edited checkpoint, GT UNMASKED
                     (hypothesis: does NOT fix ghosting — photographs stale)
  C5 ours          : locally-invalidated edited cache (masks at warp)
Reference rows: ORIG = the banked original-scene ECR output (from quals
dumps where available, else rendered via the original cache) and the real
test GT for the unaffected region.

Regions per test view (from the ORIGINAL checkpoint's rend_ids, poses only):
  R = edit region (deleted-face pixels, 8 px dilation)
  U = complement of the 16 px dilation
Metrics per protocol; paired bootstrap 10k seed 0; everything banked to
analysis/edit_aware/<scene>/.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
G1 = "/data/peilincai/gems_stage1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--original-checkpoint", required=True)
    ap.add_argument("--edited-checkpoint", required=True)
    ap.add_argument("--original-cache", required=True)
    ap.add_argument("--edited-cache", required=True)
    ap.add_argument("--rebuild-cache", default=None,
                    help="C4 cache (full rebuild, unmasked GT); skipped if absent")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from tools.gems.scenes import SCENES
    from tools.gems.eval_context import build_eval_context
    from tools.ecr.renderer import EcrRenderer
    from tools.gems.paired_bootstrap import summarize_pair
    from utils.image_utils import psnr
    from run_eval import pose_primitives

    spec = json.load(open(args.spec))
    deleted = torch.from_numpy(np.load(spec["deleted_ids_npy"])).cuda()
    os.makedirs(args.out, exist_ok=True)
    panels = os.path.join(args.out, "panels")
    os.makedirs(panels, exist_ok=True)
    quant = lambda img: img.mul(255).add(0.5).clamp_(0, 255).to(
        torch.uint8).float().div(255.0)

    def masked_mean_abs(a, b, m):
        d = (a - b).abs().mean(0)
        return float((d * m).sum() / m.sum().clamp(min=1))

    def masked_psnr(a, b, m):
        d2 = ((a - b) ** 2).mean(0)
        mse = float((d2 * m).sum() / m.sum().clamp(min=1))
        return 10.0 * np.log10(1.0 / max(mse, 1e-10))

    # ---- pass A: ORIGINAL checkpoint on test poses -> regions + ORIG-ECR ----
    ctx = build_eval_context(args.original_checkpoint, SCENES[args.scene])
    ecr_orig = EcrRenderer(args.original_cache)
    regions, orig_ecr, orig_base, gts, poses, depths = {}, {}, {}, {}, {}, {}
    with torch.no_grad():
        for cam in ctx.test_cams:
            name = str(cam.image_name)
            pkg = ctx.render_view(cam)
            ids = pkg["rend_ids"].detach().float()
            if ids.ndim == 3:
                ids = ids.unsqueeze(0)
            h, w = int(cam.image_height), int(cam.image_width)
            ids = F.interpolate(ids, size=(h, w), mode="nearest") \
                .squeeze().long().cuda()
            R = torch.isin(ids, deleted).float()[None, None]
            R8 = F.max_pool2d(R, 17, stride=1, padding=8).squeeze()
            U = 1.0 - F.max_pool2d(R, 33, stride=1, padding=16).squeeze()
            regions[name] = (R8, U)
            base = quant(pkg["render"][:3])
            orig_base[name] = base.cpu()
            adapted, _ = ecr_orig.adapt(name, pose_primitives(cam), base,
                                        pkg["surf_depth"][0])
            orig_ecr[name] = quant(adapted).cpu()
            gts[name] = quant(cam.original_image[:3].cuda().clamp(0, 1)).cpu()
            poses[name] = pose_primitives(cam)
    del ctx, ecr_orig
    torch.cuda.empty_cache()

    # ---- pass B: EDITED checkpoint -> C1 base + per-method ECR ----
    ctx = build_eval_context(args.edited_checkpoint, SCENES[args.scene])
    methods = {"C2_stale": args.original_cache, "C5_ours": args.edited_cache}
    if args.rebuild_cache and os.path.exists(
            os.path.join(args.rebuild_cache, "manifest.json")):
        methods["C4_rebuild"] = args.rebuild_cache
    renders = {m: {} for m in methods}
    c1 = {}
    with torch.no_grad():
        for cam in ctx.test_cams:
            name = str(cam.image_name)
            pkg = ctx.render_view(cam)
            c1[name] = quant(pkg["render"][:3]).cpu()
            depths[name] = pkg["surf_depth"][0].cpu()
        for m, cache in methods.items():
            ecr = EcrRenderer(cache)
            for cam in ctx.test_cams:
                name = str(cam.image_name)
                adapted, _ = ecr.adapt(
                    name, poses[name], c1[name].cuda(),
                    depths[name].cuda())
                renders[m][name] = quant(adapted).cpu()
            del ecr
            torch.cuda.empty_cache()
    del ctx
    torch.cuda.empty_cache()

    # ---- metrics ----
    names = sorted(regions)
    per_view = {"names": names}
    outs = {"C1_editedbase": c1, **renders}
    arrays = {}
    for m, imgs in outs.items():
        gr, lk, pu = [], [], []
        for n in names:
            R8, U = regions[n]
            R8c, Uc = R8.cpu(), U.cpu()
            gr.append(masked_psnr(imgs[n], orig_ecr[n], R8c))   # ghost
            lk.append(masked_mean_abs(imgs[n], c1[n], R8c))     # leak
            pu.append(masked_psnr(imgs[n], gts[n], Uc))         # true-GT U
        arrays[m] = {"ghost_psnr_R": gr, "leak_R": lk, "psnr_U_gt": pu}
    # original-ECR preservation reference on U
    pu_orig = [masked_psnr(orig_ecr[n], gts[n], regions[n][1].cpu())
               for n in names]
    arrays["ORIG_ecr"] = {"psnr_U_gt": pu_orig}

    def ci(a, b):
        return summarize_pair(np.array(a), np.array(b), floor=0.10)

    report = {
        "scene": args.scene, "spec": spec["edit_type"],
        "n_faces_deleted": spec["n_faces_deleted"],
        "n_views": len(names),
        "region_px_mean": float(np.mean(
            [regions[n][0].sum().item() for n in names])),
        "per_method": {m: {k: float(np.mean(v)) for k, v in d.items()}
                       for m, d in arrays.items()},
        "cis": {
            "ghost_C2_minus_C1": ci(arrays["C2_stale"]["ghost_psnr_R"],
                                    arrays["C1_editedbase"]["ghost_psnr_R"]),
            "ghost_C5_minus_C1": ci(arrays["C5_ours"]["ghost_psnr_R"],
                                    arrays["C1_editedbase"]["ghost_psnr_R"]),
            "presU_C5_minus_ORIG": ci(arrays["C5_ours"]["psnr_U_gt"],
                                      arrays["ORIG_ecr"]["psnr_U_gt"]),
        },
        "per_view": {m: d for m, d in arrays.items()},
    }
    if "C4_rebuild" in arrays:
        report["cis"]["ghost_C4_minus_C1"] = ci(
            arrays["C4_rebuild"]["ghost_psnr_R"],
            arrays["C1_editedbase"]["ghost_psnr_R"])
    # update-cost block from the edited cache manifest
    man = json.load(open(os.path.join(args.edited_cache, "manifest.json")))
    report["update_cost"] = man.get("update_cost", {})

    with open(os.path.join(args.out, "edit_eval.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    md = [f"# Edit-aware ECR eval — {args.scene} ({spec['edit_type']}, "
          f"{spec['n_faces_deleted']} faces)", "",
          "ghost_psnr_R: similarity of the edit region to the ORIGINAL "
          "unedited ECR output — HIGHER = MORE stale-object ghosting. "
          "leak_R: mean|. − edited-base| in-region. psnr_U_gt: TRUE-GT "
          "PSNR outside the (dilated) edit region.", "",
          "| method | ghost_psnr_R ↓ | leak_R ↓ | psnr_U (true GT) ↑ |",
          "|---|---|---|---|"]
    order = ["C1_editedbase", "C2_stale"] + \
        (["C4_rebuild"] if "C4_rebuild" in arrays else []) + \
        ["C5_ours", "ORIG_ecr"]
    for m in order:
        d = report["per_method"][m]
        md.append(f"| {m} | {d.get('ghost_psnr_R', float('nan')):.3f} "
                  f"| {d.get('leak_R', float('nan')):.4f} "
                  f"| {d.get('psnr_U_gt', float('nan')):.3f} |")
    md += ["", "**Paired CIs (10k, seed 0):**"]
    for k, v in report["cis"].items():
        md.append(f"- {k}: {v['mean_diff']:+.3f} "
                  f"[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]")
    md += ["", f"Update cost: {json.dumps(report['update_cost'])}"]
    with open(os.path.join(args.out, "edit_eval.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- panels (3 views: max/median/min region size) ----
    sizes = [(regions[n][0].sum().item(), n) for n in names]
    sizes.sort()
    picks = [sizes[-1][1], sizes[len(sizes) // 2][1], sizes[0][1]]
    for n in picks:
        row = [orig_ecr[n], c1[n], renders["C2_stale"][n]]
        if "C4_rebuild" in renders:
            row.append(renders["C4_rebuild"][n])
        row.append(renders["C5_ours"][n])
        strip = torch.cat(row, dim=2)
        torchvision.utils.save_image(strip, os.path.join(panels, f"{n}.png"))
    print(f"panels: {picks}")


if __name__ == "__main__":
    main()
