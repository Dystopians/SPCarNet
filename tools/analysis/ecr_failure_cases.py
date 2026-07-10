#!/usr/bin/env python
"""GEMS Stage-4 E9-style failure-case selection for the ECR transport
(prompt §5: occlusion errors, seam cases, coverage-gap views).

Frozen selection rules (per scene with banked final row + quals dumps):
  FC-a coverage-gap : argmin per-view covered_fraction (banked l4 row)
  FC-b transport-worst: argmin per-view dPSNR(final - base) (quals summary;
       ties to the banked rows by construction — the dump re-measures both)
  plus the count/list of ALL transport-negative views (dPSNR < 0).
Emits analysis/final_stack/ecr_failure_cases.{md,json}; the narrative
characterization (occlusion vs seam vs coverage) is written by inspecting
the dumped planes and recorded in the md by hand-edit ON TOP of the
generated skeleton (edits logged in LEDGER, numbers never touched).
"""
import json
import os

G1 = "/data/peilincai/gems_stage1"
SCENES = ["garden", "bicycle", "bonsai", "treehill", "kitchen"]


def main():
    out_dir = os.path.join(G1, "analysis", "final_stack")
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    md = ["# ECR transport — E9-style failure cases (frozen selection rules)",
          "",
          "| case | scene | view | type | covered_fraction | dPSNR "
          "(final-base) | planes |", "|---|---|---|---|---|---|---|"]
    case_no = 0
    for scene in SCENES:
        row_path = os.path.join(
            G1, "eval", f"l4_{scene}_cleanfixed30k_routed_v1", "metrics.json")
        quals_path = os.path.join(
            G1, "analysis", "quals", f"{scene}_final", "summary.json")
        if not (os.path.exists(row_path) and os.path.exists(quals_path)):
            md.append(f"| — | {scene} | PENDING | | | | |")
            continue
        row = json.load(open(row_path))
        quals = json.load(open(quals_path))["views"]
        cf = {v["image_name"]: v["covered_fraction"]
              for v in row["ecr"]["per_view"]}
        dps = {n: v["dpsnr"] for n, v in quals.items()}

        fc_a = min(cf, key=cf.get)
        fc_b = min(dps, key=dps.get)
        negatives = sorted([n for n, d in dps.items() if d < 0],
                           key=lambda n: dps[n])
        entry = {
            "coverage_gap": {"view": fc_a, "covered_fraction": cf[fc_a],
                             "dpsnr": dps.get(fc_a)},
            "transport_worst": {"view": fc_b, "dpsnr": dps[fc_b],
                                "covered_fraction": cf.get(fc_b)},
            "transport_negative_views": {n: dps[n] for n in negatives},
            "n_views": len(dps),
        }
        report[scene] = entry
        for typ, view in (("coverage-gap", fc_a), ("transport-worst", fc_b)):
            case_no += 1
            md.append(
                f"| FC{case_no:02d} | {scene} | {view} | {typ} "
                f"| {cf.get(view, float('nan')):.3f} "
                f"| {dps.get(view, float('nan')):+.3f} "
                f"| `analysis/quals/{scene}_final/{view}/` |")
        md.append(
            f"| — | {scene} | ({len(negatives)}/{len(dps)} views "
            f"transport-negative: "
            f"{', '.join(negatives[:3]) if negatives else 'none'}"
            f"{'…' if len(negatives) > 3 else ''}) | summary | | | |")
    narratives = {
        ("garden", "DSC07988"):
            "Garden's weakest coverage view is still 92% covered and "
            "IMPROVES +0.35 dB — no failure mode present; kept as the "
            "scene's honest worst case.",
        ("bicycle", "_DSC8784"):
            "COVERAGE GAP (the archetype): a low viewpoint whose entire "
            "foreground ground plane (lower half-frame) has ZERO warp "
            "support — the training trajectory never observes it from a "
            "compatible pose (conf.png: near-black lower half). The "
            "confidence gate withholds the transport there (β·valid = 0, "
            "base passes through untouched) while the supported upper half "
            "(bench/bicycle/vegetation) is corrected: net +0.13 dB at 47% "
            "coverage. Graceful degradation, no hallucination.",
        ("bonsai", "DSCF5813"):
            "Lowest-coverage bonsai view still gains +2.56 dB — the indoor "
            "ring trajectory keeps even the worst view well supported.",
        ("bonsai", "DSCF5789"):
            "Scene-worst transport delta is +0.85 dB (still strongly "
            "positive); no failure mode.",
        ("treehill", "_DSC8898"):
            "COVERAGE GAP: 47% covered (frame edges + near ground "
            "unsupported); transport corrects the covered remainder for "
            "+0.66 dB.",
        ("kitchen", "DSCF0760"):
            "Lowest-coverage kitchen view is 94% covered and gains "
            "+4.52 dB — no failure mode in this indoor ring scene.",
        ("kitchen", "DSCF0728"):
            "Scene-worst transport delta is +0.69 dB (strongly positive); "
            "the view where the base render is already best (29.1 dB).",
        ("treehill", "_DSC8946"):
            "OCCLUSION SEAM — the ONLY transport-negative view in all "
            "dumped full9 views (−0.06 dB): a close-up of the tree "
            "trunk with strong parallax; the trunk boundary shows a "
            "zero-confidence occlusion seam (conf.png: dark crack right of "
            "the trunk) and the bench/near-ground is largely unsupported, "
            "so the transport can only act on the trunk and the distant "
            "band. Residual error stays in the high-frequency background "
            "(err_final.png); the confidence gate bounds the damage to "
            "-0.06 dB rather than corrupting the frame.",
    }
    md += ["", "Types are SELECTION rules; the occlusion/seam/coverage "
           "characterization per case comes from inspecting the dumped "
           "planes (base/final/err/conf/count/beta):", "",
           "## Case narratives", ""]
    for (scene, view), text in narratives.items():
        if scene in report and any(
                view == c.get("view") for c in
                (report[scene]["coverage_gap"],
                 report[scene]["transport_worst"])):
            md.append(f"- **{scene} / {view}** — {text}")
    total = sum(e["n_views"] for e in report.values())
    n_neg = sum(len(e["transport_negative_views"]) for e in report.values())
    md += ["", f"**Headline:** across all {total} dumped full9 test views, "
           f"the routed transport is PSNR-positive on every view but "
           f"{n_neg} (−0.06 dB, occlusion-seam case above); coverage gaps "
           "degrade gracefully to the base render via the STRUCTURAL "
           "compose gate (β·valid — see GOAL #E-08: the certification "
           "lives in the valid mask, not the net's confidence inputs) "
           "instead of hallucinating — that is what makes the worst case "
           "boring."]
    with open(os.path.join(out_dir, "ecr_failure_cases.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "ecr_failure_cases.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
