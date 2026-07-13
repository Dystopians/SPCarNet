"""GEMS B3 (QEM+FT) column analysis — LEDGER GOAL #013.

Reads banked metrics.json rows (single mouth, D5) for clean / B2 / B4 / B5 and
the new B3 rows at B50 on {garden, toy_parking, courtyard}; emits the 3-scene
table with paired bootstrap CIs (PROTOCOL section 5: seed 0, 10k resamples)
and the pre-registered verdict:

    PREDICTION (GOAL #013): B3 < B5 on all 3 scenes; margin >= +0.10 dB
    (B5-B3, CI excl. 0) on >= 2/3 scenes.

Usage: $PY -m tools.gems.b3_analysis
Writes /data/peilincai/gems_stage1/analysis/b3_qem/{b3_table.md, b3_summary.json}
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from tools.gems.paired_bootstrap import paired_bootstrap_ci  # noqa: E402

EVAL_ROOT = "/data/peilincai/gems_stage1/eval"
OUT_DIR = "/data/peilincai/gems_stage1/analysis/b3_qem"

ROWS = {
    "garden": {
        "clean": "garden_clean30k_v2",
        "B2_random_ft": "garden_B50_random_ft_e1b",
        "B4_evidence_noft": "garden_B50_importance_noft_e1b",
        "B5_gems_core": "garden_B50_importance_ft_e1v2",
        "B3_qem_ft": "garden_B50_qem_ft_b3",
    },
    "toy_parking": {
        "clean": "toy_parking_clean30k_v1",
        "B2_random_ft": "toy_parking_B50_random_ft_e1b",
        "B4_evidence_noft": "toy_parking_B50_importance_noft_e1b",
        "B5_gems_core": "toy_parking_B50_importance_ft_e1v2",
        "B3_qem_ft": "toy_parking_B50_qem_ft_b3",
    },
    "courtyard": {
        "clean": "courtyard_clean30k_v1",
        "B2_random_ft": "courtyard_B50_random_ft_e1b",
        "B4_evidence_noft": "courtyard_B50_importance_noft_e1b",
        "B5_gems_core": "courtyard_B50_importance_ft_e1v2",
        "B3_qem_ft": "courtyard_B50_qem_ft_b3",
    },
}


def load(row_name: str) -> dict:
    p = os.path.join(EVAL_ROOT, row_name, "metrics.json")
    with open(p) as f:
        m = json.load(f)
    m["_path"] = p
    return m


def pv(m: dict, key: str) -> np.ndarray:
    return np.asarray(m["rendering"]["per_view"][key], dtype=np.float64)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    summary: dict = {"ledger_goal": "GOAL #013", "budget": "B50",
                     "protocol": "paired bootstrap seed 0, 10k resamples "
                                 "(tools/gems/paired_bootstrap.py)",
                     "scenes": {}}
    lines = ["# B3 (QEM decimation + safe FT) @ B50 — 3-scene table "
             "(LEDGER GOAL #013)", ""]
    n_scenes_margin_met = 0
    b3_below_b5_all = True

    for scene, rows in ROWS.items():
        ms = {k: load(v) for k, v in rows.items()}
        names0 = ms["clean"]["rendering"]["per_view"]["image_names"]
        for k, m in ms.items():
            assert m["rendering"]["per_view"]["image_names"] == names0, (
                f"{scene}/{k}: per-view image_names differ from clean — "
                "pairing invalid")
        t_clean = ms["clean"]["cost"]["n_triangles"]
        lines += [f"## {scene} (clean T={t_clean:,}; n_test={len(names0)})",
                  "",
                  "| method | tris (frac of clean) | PSNR | SSIM | LPIPS | FPS "
                  "| dPSNR vs clean [95% CI] | dLPIPS vs clean [95% CI] |",
                  "|---|---|---|---|---|---|---|---|"]
        srec: dict = {"rows": {}, "pairwise_vs_B3": {}}
        for k, m in ms.items():
            r = m["rendering"]["mean"]
            tri = m["cost"]["n_triangles"]
            if k == "clean":
                dp = dl = "—"
            else:
                cp = paired_bootstrap_ci(pv(m, "psnr"), pv(ms["clean"], "psnr"))
                cl = paired_bootstrap_ci(pv(m, "lpips"), pv(ms["clean"], "lpips"))
                dp = (f"{cp['mean_diff']:+.3f} [{cp['ci_lo']:+.3f},"
                      f"{cp['ci_hi']:+.3f}]")
                dl = (f"{cl['mean_diff']:+.4f} [{cl['ci_lo']:+.4f},"
                      f"{cl['ci_hi']:+.4f}]")
                srec["rows"].setdefault(k, {})["dpsnr_vs_clean"] = cp
                srec["rows"].setdefault(k, {})["dlpips_vs_clean"] = cl
            srec["rows"].setdefault(k, {}).update({
                "eval_row": rows[k], "metrics_json": m["_path"],
                "n_triangles": tri, "psnr": r["psnr"], "ssim": r["ssim"],
                "lpips": r["lpips"], "fps": m["cost"]["render_fps"]})
            lines.append(
                f"| {k} | {tri:,} ({tri / t_clean:.3f}) | {r['psnr']:.3f} | "
                f"{r['ssim']:.4f} | {r['lpips']:.4f} | "
                f"{m['cost']['render_fps']:.1f} | {dp} | {dl} |")

        # fair-budget check across the four B50 rows
        b50 = [srec["rows"][k]["n_triangles"] for k in
               ("B2_random_ft", "B4_evidence_noft", "B5_gems_core", "B3_qem_ft")]
        spread = (max(b50) - min(b50)) / float(min(b50))
        srec["fair_budget_max_rel_spread"] = spread
        assert spread <= 0.02, f"{scene}: budget spread {spread:.4f} > 2%"

        # pre-registered pairwise: B3 vs B5 / B4 / B2 (PSNR + LPIPS)
        lines += ["", "| pairwise (B3 − X) | dPSNR [95% CI] | dLPIPS [95% CI] |",
                  "|---|---|---|"]
        for other in ("B5_gems_core", "B4_evidence_noft", "B2_random_ft"):
            cp = paired_bootstrap_ci(pv(ms["B3_qem_ft"], "psnr"),
                                     pv(ms[other], "psnr"))
            cl = paired_bootstrap_ci(pv(ms["B3_qem_ft"], "lpips"),
                                     pv(ms[other], "lpips"))
            srec["pairwise_vs_B3"][other] = {"dpsnr": cp, "dlpips": cl}
            lines.append(
                f"| B3 − {other} | {cp['mean_diff']:+.3f} "
                f"[{cp['ci_lo']:+.3f},{cp['ci_hi']:+.3f}] | "
                f"{cl['mean_diff']:+.4f} [{cl['ci_lo']:+.4f},"
                f"{cl['ci_hi']:+.4f}] |")
        b5 = srec["pairwise_vs_B3"]["B5_gems_core"]["dpsnr"]
        margin_met = (b5["mean_diff"] <= -0.10) and (b5["ci_hi"] < 0.0)
        below = b5["mean_diff"] < 0.0
        n_scenes_margin_met += int(margin_met)
        b3_below_b5_all &= below
        srec["b3_below_b5"] = below
        srec["b5_margin_ge_0p10_ci_excl0"] = margin_met
        lines += ["", f"B3 below B5: {below}; margin >= 0.10 dB with CI excl. "
                      f"0: {margin_met}", ""]
        summary["scenes"][scene] = srec

    verdict = {
        "prediction": "B3 < B5 on all 3 scenes; B5-B3 >= +0.10 dB CI excl. 0 "
                      "on >= 2/3 scenes",
        "b3_below_b5_all_scenes": b3_below_b5_all,
        "n_scenes_margin_met": n_scenes_margin_met,
        "prediction_met": bool(b3_below_b5_all and n_scenes_margin_met >= 2),
    }
    summary["verdict"] = verdict
    lines += ["## Verdict", "", f"```json\n{json.dumps(verdict, indent=1)}\n```", ""]

    with open(os.path.join(OUT_DIR, "b3_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(OUT_DIR, "b3_table.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"[b3] wrote {OUT_DIR}/b3_table.md and b3_summary.json")


if __name__ == "__main__":
    main()
