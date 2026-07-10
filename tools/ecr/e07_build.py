#!/usr/bin/env python
"""GEMS Stage-4 GOAL #E-07: matched-TOTAL-storage 3DGS rerun (prompt §3 L5 / §4b).

Applies the R1 storage-match rule with the NEW budget = ECR final-stack TOTAL
artifact MB (checkpoint + raw evidence cache) on the R1 trio. Per the
pre-registration (LEDGER GOAL #E-06 block): where vanilla 3DGS is already
under the target, it is reported as-is (R1 kitchen precedent) — pruning can
only shrink, and growing past the stock 30k recipe is not a public recipe
point. All numbers come from banked artifacts (R1 per-scene jsons from GOAL
#017 + banked final-stack metrics.json rows); nothing is hand-typed and no
new training is run unless a scene's vanilla ply exceeds its target (in which
case this script FAILS loudly so the prune+FT chain can be launched).

Same sanctioned single-mouth exception + context-only status as R1 (GOAL
#017): cross-representation, no CIs, gates nothing.
"""
import json
import os

G1 = "/data/peilincai/gems_stage1"
SCENES = ["garden", "bicycle", "kitchen"]
R1_DIR = os.path.join(G1, "analysis", "r1_3dgs_reference")
OUT_DIR = os.path.join(G1, "analysis", "final_stack")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"goal": "GOAL #E-07", "cell": "matched-TOTAL-storage 3DGS rerun",
              "context_only": True, "scenes": {}}
    md = ["# GOAL #E-07 — 3DGS at matched TOTAL artifact storage "
          "(checkpoint + evidence cache)", "",
          "> Rerun of the R1 comparison (GOAL #017) with the honest Stage-4 "
          "budget: the ECR final stack ships checkpoint + raw evidence cache, "
          "so 3DGS is granted that TOTAL as its storage target. "
          "Match rule unchanged (opacity-prune to target; vanilla reported "
          "as-is when already under target — R1 kitchen precedent). "
          "CONTEXT-ONLY cross-representation reference, no CIs, gates "
          "nothing; same sanctioned single-mouth exception as R1 "
          "(r1_metrics.py mirrors run_eval conventions exactly).", "",
          "| scene | point | PSNR | SSIM | LPIPS | artifact MB | FPS |",
          "|---|---|---|---|---|---|---|"]
    for scene in SCENES:
        r1 = json.load(open(os.path.join(R1_DIR, f"r1_{scene}.json")))
        fin = json.load(open(os.path.join(
            G1, "eval", f"l4_{scene}_cleanfixed30k_routed_v1",
            "metrics.json")))
        van = r1["rows"]["3dgs_30k"]
        cost = fin["cost"]
        target_mb = cost["total_artifact_mb"]
        vanilla_mb = van["disk_mb"]
        if vanilla_mb > target_mb:
            raise SystemExit(
                f"{scene}: vanilla 3DGS ({vanilla_mb:.1f} MB) EXCEEDS the "
                f"TOTAL target ({target_mb:.1f} MB) — launch the prune+FT "
                "chain for this scene instead of reporting as-is.")
        fm = fin["rendering"]["mean"]
        report["scenes"][scene] = {
            "target_total_mb": target_mb,
            "ecr_final_stack": {
                "row": f"l4_{scene}_cleanfixed30k_routed_v1",
                "psnr": fm["psnr"], "ssim": fm["ssim"], "lpips": fm["lpips"],
                "ckpt_mb": cost["disk_mb"],
                "cache_mb_raw": cost["cache_mb_raw"],
                "cache_mb_compressed": cost["cache_mb_compressed"],
                "total_mb": target_mb,
                "end_to_end_fps": cost.get("end_to_end_fps"),
            },
            "3dgs_matched_total": {
                "source": f"r1_{scene}.json rows.3dgs_30k (GOAL #017, "
                          "banked; vanilla <= target => as-is)",
                "psnr": van["psnr"], "ssim": van["ssim"],
                "lpips": van["lpips"], "disk_mb": vanilla_mb,
                "fps": van["render_fps"], "n_gaussians": van["primitives"],
                "storage_headroom_mb": target_mb - vanilla_mb,
                "pruned": False,
            },
            "gap": {"dpsnr_3dgs_minus_ecr": van["psnr"] - fm["psnr"],
                    "dlpips_3dgs_minus_ecr": van["lpips"] - fm["lpips"]},
        }
        e = report["scenes"][scene]
        md.append(
            f"| {scene} | ECR final stack (v3 routed) | {fm['psnr']:.3f} "
            f"| {fm['ssim']:.4f} | {fm['lpips']:.4f} "
            f"| {target_mb:.0f} (ckpt {cost['disk_mb']:.0f} + cache "
            f"{cost['cache_mb_raw']:.0f}) "
            f"| {cost.get('end_to_end_fps', -1):.2f} (e2e) |")
        md.append(
            f"| {scene} | 3DGS @ matched TOTAL (vanilla as-is, "
            f"{100 * vanilla_mb / target_mb:.0f}% of budget) "
            f"| {van['psnr']:.3f} | {van['ssim']:.4f} | {van['lpips']:.4f} "
            f"| {vanilla_mb:.0f} | {van['render_fps']:.1f} |")
    # honest positioning paragraph, computed from the banked comparisons
    new_gaps = [report["scenes"][s]["gap"]["dpsnr_3dgs_minus_ecr"]
                for s in SCENES]
    budget_pcts = [100 * report["scenes"][s]["3dgs_matched_total"]["disk_mb"]
                   / report["scenes"][s]["target_total_mb"] for s in SCENES]
    md += ["",
           "**Reading (context, no CIs):** even granted the full ECR TOTAL "
           "budget, stock 3DGS-30k uses only "
           + ", ".join(f"{p:.0f}%" for p in budget_pcts)
           + " of it on {garden, bicycle, kitchen} and remains ahead on PSNR "
           "by " + ", ".join(f"{g:+.2f}" for g in new_gaps)
           + " dB respectively. "
           "The Stage-2 R1 comparison (vs GEMS B5@B50 alone) had 3DGS ahead "
           "by 2.1–3.4 dB; the ECR final stack closes most of that "
           "cross-representation gap while shipping a triangle-mesh artifact "
           "plus a train-view evidence cache — the axes on which the mesh "
           "artifact is the deliverable (downstream geometry cells, "
           "compaction exactness) are unchanged from the R1 A3 paragraph.",
           "", "_Generated by tools/ecr/e07_build.py from banked artifacts "
           "only (r1_<scene>.json + l4_<scene>_cleanfixed30k_routed_v1)._"]
    with open(os.path.join(OUT_DIR, "e07_matched_total_3dgs.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_DIR, "e07_matched_total_3dgs.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n".join(md))
    print(f"\nwrote {os.path.join(OUT_DIR, 'e07_matched_total_3dgs.md')}")


if __name__ == "__main__":
    main()
