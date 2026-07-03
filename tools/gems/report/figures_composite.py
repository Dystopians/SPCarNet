#!/usr/bin/env python
"""GEMS Stage-2 evidence pack — composite figures F3 / F6 / F8 (section 8).

  F3  method/pipeline diagram DRAFT (boxes + caption text file; humans polish).
      Main flow: checkpoint -> evidence -> prune -> safe-FT -> single-mouth
      eval; the consumption trilogy + evidence-vs-error appear as an ANALYSIS
      branch (section 7.5 framing: analysis, not method).
  F6  downstream figure: occupancy confusion (d1, from the corpus) + the
      N=500 planner table (script-read from analysis/e5_down_ext/summary.json)
      + maneuver example panels (copied from the banked R3.c/R3.b panels —
      no recomputation).
  F8  failure board: contact sheet of the 13 E9 taxonomy cases with one-line
      captions (panels copied from analysis/e9_failure_taxonomy/, no
      recomputation; full diagnoses live in TAXONOMY.md).

All numbers rendered in F6 are read from durable artifacts (all_rows.json /
summary.json); none hand-typed. Caption .txt files accompany F3 and F6.

Usage:
    python tools/gems/report/figures_composite.py [--rows FILE] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402

from tools.gems.report.tables import Corpus, ALL_ROWS_DEFAULT  # noqa: E402
from tools.gems.report.figures import (  # noqa: E402
    C_B5, C_B4, C_B2, C_B6R, C_B3, INK, INK2, GRID, SURFACE)

OUTDIR_DEFAULT = os.path.join(REPO_ROOT, "RESULTS", "figures")
ANALYSIS_ROOT = "/data/peilincai/gems_stage1/analysis"
E5_SUMMARY = os.path.join(ANALYSIS_ROOT, "e5_down_ext", "summary.json")
R3C_PANELS = os.path.join(ANALYSIS_ROOT, "r3c_planner", "panels")
R3B_PANELS = os.path.join(ANALYSIS_ROOT, "r3b_submesh", "panels")
E9_DIR = os.path.join(ANALYSIS_ROOT, "e9_failure_taxonomy")


# ---------------------------------------------------------------------------
# F3 — pipeline diagram DRAFT
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc, ec=INK2, fs=8.5, style="round,pad=0.02",
         text_color=INK, lw=1.2, ls="solid"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                       edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, zorder=3, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color=INK, ls="solid", lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw, color=color,
                                 linestyle=ls, zorder=4))


def f3_pipeline(outdir):
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    # ---- main pipeline row (y = 32..44) ----
    main_y, bh, bw = 33, 11, 16
    xs = [2, 22, 42, 62, 82]
    _box(ax, xs[0], main_y, bw, bh,
         "clean MeshSplatting\ncheckpoint (B0)\n30k iters, frozen",
         "#eef3fb")
    _box(ax, xs[1], main_y, bw, bh,
         "EVIDENCE\ntrain-view render pass\nper-triangle pixels_total\n(D4: train views only)",
         "#e8f7f0")
    _box(ax, xs[2], main_y, bw, bh,
         "PRUNE to budget B\nkeep top-K by evidence\nK = floor(B·T), ±2% fair\nbudget across methods",
         "#fdf3dc")
    _box(ax, xs[3], main_y, bw, bh,
         "SAFE FINE-TUNE\nfeatures-only, 10k iters\npositions/weights FROZEN\n(drift-safe channel)",
         "#f3effc")
    _box(ax, xs[4], main_y, bw, bh,
         "compact checkpoint\n(B5 GEMS-core)\nplain mesh artifact,\nno test-time extras",
         "#eef3fb")
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + bw, main_y + bh / 2, b, main_y + bh / 2)

    # ---- eval mouth (below the compact checkpoint) ----
    _box(ax, 62, 16, 36, 10,
         "SINGLE-MOUTH EVAL  (run_eval.py, PROTOCOL v1.1.x)\n"
         "PSNR/SSIM/LPIPS + g1–g4 geometry + d1/d2 downstream\n"
         "+ FPS/disk/VRAM; paired per-view bootstrap CIs (seed 0, 10k)",
         "#fbecec")
    _arrow(ax, xs[4] + bw / 2, main_y, 85, 26)
    ax.text(62, 27.3, "anchors compared in the same mouth: clean@30k (legacy) "
            "+ clean-fixed@30k (primary)", fontsize=7.3, color=INK2)

    # ---- analysis branch (dashed frame, bottom-left) ----
    _box(ax, 2, 2, 56, 22, "", SURFACE, ec=INK2, ls=(0, (5, 4)), lw=1.2)
    ax.text(4, 21.6, "ANALYSIS BRANCH — analysis, not method (§7.5 framing)",
            fontsize=8.5, color=INK, fontweight="bold")
    ay, ah, aw = 9.5, 8.5, 12.4
    _box(ax, 4, ay, aw, ah,
         "R3.a occupancy\nroutes: voxelize\nvs TSDF fusion\n(TSDF falsified)",
         "#f4f4f2", fs=7.3)
    _box(ax, 17.8, ay, aw, ah,
         "R3.c planner loop\nHybrid-A*-lite,\nN=100→500 paired\n(raw grids unusable)",
         "#f4f4f2", fs=7.3)
    _box(ax, 31.6, ay, aw, ah,
         "R3.b certified\nsub-mesh\n(sheds load-bearing\nsurface; falsified)",
         "#f4f4f2", fs=7.3)
    _box(ax, 45.4, ay, aw, ah,
         "evidence-vs-error\nρ≈0.7 (3/3 scenes);\nsilent in coverage\ngaps (5–11× tail)",
         "#f4f4f2", fs=7.3)
    for x0 in (4 + aw, 17.8 + aw, 31.6 + aw):
        _arrow(ax, x0, ay + ah / 2, x0 + 1.4, ay + ah / 2, color=INK2, lw=1.1)
    ax.text(4, 4.6, "consumption trilogy: one-time artifacts from TRAIN evidence only; "
                    "verdicts are citable negatives with mechanisms (C4′(2))",
            fontsize=7.3, color=INK2)
    # branch input arrows: evidence + compact ckpt feed the analysis
    _arrow(ax, xs[1] + bw / 2, main_y, xs[1] + bw / 2 - 6, 24,
           color=INK2, ls=(0, (4, 3)))
    _arrow(ax, xs[3] + bw / 2, main_y, 50, 24, color=INK2, ls=(0, (4, 3)))
    ax.text(51.5, 27.8, "checkpoints +\ntrain evidence", fontsize=6.8,
            color=INK2, ha="center")

    # ---- demoted axes note (bottom-right) ----
    _box(ax, 62, 2, 36, 10,
         "DEMOTED / TOMBSTONED (train-time only, default-off):\n"
         "geometry losses (E2, 3/3 FAIL) · teacher distillation (E3,\n"
         "sunset; train-only) · B6R opacity release (bounded positive)",
         "#f7f7f5", ec=INK2, fs=7.6, ls=(0, (2, 2)))

    fig.suptitle("F3 (DRAFT) — GEMS pipeline: evidence-guided compaction "
                 "under explicit budgets + drift-safe fine-tuning;\n"
                 "analysis branch separated by design",
                 fontsize=11, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"F3_pipeline_draft.{ext}"), dpi=200)
    plt.close(fig)
    print("[figures_composite] F3_pipeline_draft.png/.pdf")

    with open(os.path.join(outdir, "F3_caption.txt"), "w") as f:
        f.write(
            "F3 (draft caption): GEMS pipeline. From a frozen MeshSplatting "
            "checkpoint, a single train-view render pass accumulates "
            "per-triangle evidence (pixel ownership; D4-pure, train views "
            "only); triangles are pruned to an explicit budget B by evidence "
            "rank under a fair-budget rule (triangle counts within ±2% across "
            "compared methods); the survivor mesh is fine-tuned through the "
            "drift-safe channel only (features/SH; positions and weights "
            "frozen — all-parameter fine-tuning measurably destroys converged "
            "checkpoints, NEGATIVE_RESULTS §2). The output is a plain, "
            "smaller checkpoint of the SAME format: no test-time components. "
            "All numbers flow through one eval mouth (run_eval.py) with "
            "paired per-view bootstrap CIs against dual anchors. Bottom "
            "(dashed): the analysis branch — the occupancy/planner "
            "consumption trilogy and the evidence-vs-error study — consumes "
            "checkpoints and train evidence but feeds NOTHING back into the "
            "method; it is reported as analysis (Stage-2 §7.5), and its "
            "verdicts are citable negatives with mechanisms. Demoted axes "
            "(geometry losses, teacher distillation, opacity release) remain "
            "in-tree, default-off, train-time only.\n")
    print("[figures_composite] F3_caption.txt")


# ---------------------------------------------------------------------------
# F6 — downstream figure
# ---------------------------------------------------------------------------

def f6_downstream(c: Corpus, outdir):
    e5 = json.load(open(E5_SUMMARY))
    cells = e5["cells"]

    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15), hspace=0.34,
                          wspace=0.22)
    fig.patch.set_facecolor(SURFACE)

    # (a) occupancy confusion — d1 false-free / false-occupied, clean vs B50
    ax = fig.add_subplot(gs[0, 0])
    scenes = ["toy_parking", "toy_parking_v2", "toy_parking_occl", "courtyard"]
    ff_clean, ff_b50, fo_clean, fo_b50, labels = [], [], [], [], []
    for s in scenes:
        b0 = c.canon(s, "B0", "B100")
        b5 = c.canon(s, "B5", "B50")
        if not (b0 and b5 and b0.get("d1") and b5.get("d1")):
            continue
        if "false_free_rate" not in (b0["d1"] or {}):
            continue
        labels.append(s.replace("toy_parking", "toy").replace("__", "_"))
        ff_clean.append(b0["d1"]["false_free_rate"] * 100)
        ff_b50.append(b5["d1"]["false_free_rate"] * 100)
        fo_clean.append(b0["d1"]["false_occupied_rate"] * 100)
        fo_b50.append(b5["d1"]["false_occupied_rate"] * 100)
    x = np.arange(len(labels))
    w = 0.2
    ax.bar(x - 1.5 * w, ff_clean, w, color=C_B4, label="false-FREE, clean")
    ax.bar(x - 0.5 * w, ff_b50, w, color=C_B5, label="false-FREE, B50")
    ax.bar(x + 0.5 * w, fo_clean, w, color="#cfcecb", label="false-occ, clean")
    ax.bar(x + 1.5 * w, fo_b50, w, color=INK2, label="false-occ, B50")
    for xi, (a_, b_) in enumerate(zip(ff_clean, ff_b50)):
        ax.text(xi - w, max(a_, b_) + 1.2, f"{a_:.0f}/{b_:.0f}", ha="center",
                fontsize=7.2, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(max(ff_clean), max(ff_b50)) * 1.35)
    ax.set_ylabel("rate [% of GT-occupied / GT-free voxels]", fontsize=8)
    ax.set_title("(a) d1 occupancy confusion @0.10 m — B50 PRESERVES the\n"
                 "clean model's (bad) confusion; false-free is safety-critical",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # (b) N=500 planner table (script-read)
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    rows = []
    order = [("toy_parking__GTREF", "toy GT ref"),
             ("toy_parking__clean30k__route_i", "toy clean rt-i"),
             ("toy_parking__B50_importance_ft_e1v2_40000__route_i", "toy B50 rt-i"),
             ("courtyard__GTREF", "cyard GT ref"),
             ("courtyard__clean30k__route_i", "cyard clean rt-i"),
             ("courtyard__B50_importance_ft_e1v2_40000__route_i", "cyard B50 rt-i"),
             ("courtyard__clean30k__route_ii", "cyard clean rt-ii"),
             ("courtyard__B50_importance_ft_e1v2_40000__route_ii", "cyard B50 rt-ii")]
    for key, label in order:
        v = cells.get(key)
        if v is None:
            continue
        rows.append([label, f"{v['plans_found']}/500",
                     f"{v['collisions_per_100_plans']:.1f}",
                     f"{v['spurious_infeasibility_rate'] * 100:.1f}%"])
    tab = ax.table(cellText=rows,
                   colLabels=["cell", "found", "coll./100 plans",
                              "spurious infeas."],
                   loc="center", cellLoc="left", colLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.5)
    tab.auto_set_column_width(col=list(range(4)))
    for (r_, c_), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        if r_ == 0:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor("#f2f1ee")
    ax.set_title("(b) N=500 paired planner problems (seed 0; GOAL#015):\n"
                 "clean↔B50 outcome sets IDENTICAL (CIs [0,0]); raw grids are\n"
                 "almost-always infeasible AND unsafe when they do plan",
                 fontsize=9)

    # (c)+(d) maneuver example panels (banked, no recomputation)
    for i, (path, title) in enumerate((
            (os.path.join(R3C_PANELS, "courtyard__gt_collision.png"),
             "(c) maneuver example — courtyard route-ii plan sweeps a GT wall\n"
             "its grid marks free (R3.c panel; grid bit-identical clean↔B50)"),
            (os.path.join(R3B_PANELS, "courtyard__footprint_before_after.png"),
             "(d) R3.b certification sheds load-bearing surface — courtyard\n"
             "footprint before/after (kept set EXACTLY identical clean↔B50)"))):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(title, fontsize=9)

    fig.suptitle("F6 — Downstream proxies (E5-DOWN): preservation-exact under "
                 "compaction; consumption of RAW splat geometry fails with "
                 "mechanisms (C4′)", fontsize=11.5, color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"F6_downstream.{ext}"), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print("[figures_composite] F6_downstream.png/.pdf")

    with open(os.path.join(outdir, "F6_caption.txt"), "w") as f:
        f.write(
            "F6 (draft caption): Downstream proxies on GT-mesh scenes. "
            "(a) Occupancy confusion at 10 cm voxels: the clean models' "
            "false-free rates (safety-critical direction) are large and B50 "
            "compaction preserves them almost exactly — compaction neither "
            "causes nor fixes the problem (C4′ is a preservation claim; C2's "
            "measurement claim is that photometric quality masks this). "
            "(b) 500 paired parking-maneuver planning problems per scene "
            "(Hybrid-A*-lite; straight/arc/reverse primitives at a 4.5x1.8 m "
            "footprint): clean vs B50 planner outcome sets are IDENTICAL "
            "(found/collision-set CIs [0,0]); raw route-i grids are 88–100% "
            "spuriously infeasible, and courtyard's only 2/500 allowed plans "
            "BOTH hit GT geometry. (c) A courtyard route-ii plan colliding "
            "with a real wall the fused grid marks free (grazing-ray TSDF "
            "bias). (d) The R3.b certified sub-mesh sheds load-bearing "
            "surface together with junk (same train-coverage selection-effect "
            "family). ESDF context (analysis/e5_down_ext/esdf_table.md): "
            "model grids under-estimate clearance by 1.0–2.9 m mean "
            "everywhere. Blocker is baseline checkpoint geometry, not "
            "compaction; frozen fix-target: courtyard >=30/100 found at "
            "<=3.0 collisions/100 simultaneously.\n")
    print("[figures_composite] F6_caption.txt")


# ---------------------------------------------------------------------------
# F8 — failure board (contact sheet of the 13 E9 cases)
# ---------------------------------------------------------------------------

# (representative panel, one-line caption) per case — diagnoses quoted from
# analysis/e9_failure_taxonomy/TAXONOMY.md (families A-E); panels banked.
F8_CASES = [
    ("case01_flowers_B50_ft__DSC9144.png",
     "1|C flowers B50: only S-REND floor fail (-0.15); diffuse speckle, FT can't repaint"),
    ("case02_town06_B50_ft_front00000088.png",
     "2|C town06 B50 (-0.45): least over-parameterized town; far-field damage"),
    ("case03_toy_B50_noft_00035.png",
     "3|A toy B50 residual (-0.52): train-coverage gap erodes wall/ground junction"),
    ("case04_toy_e2v3_floaterprune_ft_00035.png",
     "4|A E2v3 floater deletion: -8.07 dB hole at view 00035 (selection effect)"),
    ("case05_garden_B50_defaultft_DSC08028.png",
     "5|B default FT after prune: -2.9 dB global washout (position drift)"),
    ("case06_toy_m3v1_gradrouted_00015.png",
     "6|B gradient-routed geometry losses: silhouette halos; guard blown"),
    ("case07_toy_e3v2_shdistill_00015.png",
     "7|D SH-distill on 72-view toy: LPIPS 0.11->0.21 (view-manifold overfit)"),
    ("case08_courtyard_B25_random_DSC_0302.png",
     "8|C courtyard random-B25 (-3.97): starvation without evidence ranking"),
    ("case09_toy_e2r_faded_00035.png",
     "9|A E2R fading on toy: pinholes on low-train-support content; guard blown"),
    ("case10_courtyard_r3b_footprint_before_after.png",
     "10|A/E R3.b certification sheds real walls with junk (coll. 16.7/100)"),
    ("case11_toy_r3a_tsdf_coverage_loss.png",
     "11|E R3.a TSDF grazing-ray bias: near-surface voxels vote FREE"),
    ("case12_garden_B50_noft_floaters_DSC08052.png",
     "12|E garden g3 fragmentation 1941->5724 components at iso-PSNR"),
    ("case13_kitchen_B12_ft_DSCF0768.png",
     "13|C kitchen B12.5 (-2.37): aggressive-budget starvation, indoor worst"),
]


def f8_failure_board(outdir):
    # 2-col layout: the banked panels are wide 4-up strips — give each tile
    # the full half-width and stack captions above.
    n = len(F8_CASES)
    ncol, nrow = 2, 7  # 13 tiles + 1 legend tile
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.6 * ncol, 2.75 * nrow))
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()
    for ax, (fname, caption) in zip(axes, F8_CASES):
        img = mpimg.imread(os.path.join(E9_DIR, fname))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(caption, fontsize=9.2, color=INK, loc="left", pad=4)
    # legend tile
    ax = axes[n]
    ax.axis("off")
    ax.text(0.02, 0.98,
            "E9-FAIL taxonomy — 5 mechanism families:  "
            "A train-coverage selection effects · B optimizer drift on "
            "converged ckpts ·\nC budget starvation / headroom exhaustion · "
            "D view-conditioning capacity mismatch · E baseline-geometry\n"
            "unreliability & consumption-route bias.\n\n"
            "13 cases, 24 panels; one-paragraph diagnoses, evidence pointers "
            "and 'what it bounds' lines in\n"
            "analysis/e9_failure_taxonomy/TAXONOMY.md (LEDGER GOAL#R-09).",
            fontsize=9.6, va="top", color=INK, linespacing=1.6)
    for ax in axes[n + 1:]:
        ax.axis("off")
    fig.suptitle("F8 — Failure board: 13 curated failure cases across 5 "
                 "mechanism families (E9-FAIL; captions = case|family, "
                 "one-line diagnosis)", fontsize=13, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"F8_failure_board.{ext}"), dpi=150)
    plt.close(fig)
    print("[figures_composite] F8_failure_board.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=ALL_ROWS_DEFAULT)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    with open(args.rows) as f:
        c = Corpus(json.load(f))
    os.makedirs(args.outdir, exist_ok=True)
    f3_pipeline(args.outdir)
    f6_downstream(c, args.outdir)
    f8_failure_board(args.outdir)


if __name__ == "__main__":
    main()
