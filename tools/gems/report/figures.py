#!/usr/bin/env python
"""GEMS Stage-2 evidence pack — figure generator (F1 draft, F2, F7).

All data is read from RESULTS/aggregate/all_rows.json (metrics.json-derived);
no hand-typed numbers. PNG + PDF under RESULTS/figures/.

Design: dataviz method — fixed categorical hue order (never cycled), one axis
per panel (no dual axes), thin marks, direct end-labels, recessive grid,
anchors drawn as neutral-ink reference marks (they are references, not series).

  F2  Pareto curves: PSNR & LPIPS vs triangle count, per suite, dual-anchor
      reference (B0 clean@30k dashed, B0' clean-fixed@30k solid).
  F1  teaser DRAFT (garden): budget -> quality -> geometry at one glance.
  F7  ablation chart: dPSNR vs reference with 95% CI whiskers.

Usage:
    python tools/gems/report/figures.py [--rows FILE] [--outdir DIR]
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
import numpy as np  # noqa: E402

from tools.gems.report.tables import (  # noqa: E402
    Corpus, paired_delta, BUDGETS, ALL_ROWS_DEFAULT)

OUTDIR_DEFAULT = os.path.join(REPO_ROOT, "RESULTS", "figures")

# validated categorical palette (dataviz reference instance, light mode),
# assigned in fixed order to the method series — never cycled:
C_B5 = "#2a78d6"      # slot 1 blue  — B5 GEMS-core
C_B4 = "#1baf7a"      # slot 2 aqua  — B4 prune-no-FT
C_B2 = "#eda100"      # slot 3 yellow— B2 random+FT (relief: direct labels)
C_B5I = "#008300"     # slot 4 green — B5-iter
C_B6R = "#4a3aa7"     # slot 5 violet— B6R opacity release
C_B3 = "#c8326b"      # slot 6 pink  — B3 QEM decimation + FT
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"

SERIES = [("B5", "B5 GEMS-core", C_B5),
          ("B4", "B4 prune, no FT", C_B4),
          ("B2", "B2 random + FT", C_B2),
          ("B3", "B3 QEM + FT", C_B3),
          ("B5-iter", "B5-iter (2-step)", C_B5I),
          ("B6R", "B6R opacity-release", C_B6R)]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def series_points(c: Corpus, scene, method):
    pts = []
    for b in ("B12.5", "B25", "B50"):
        r = c.canon(scene, method, b)
        if r is not None:
            pts.append((r["cost"]["n_triangles"] / 1e6,
                        r["rendering_mean"]["psnr"],
                        r["rendering_mean"]["lpips"], b))
    return pts


def draw_scene_panel(ax, c: Corpus, scene, metric_idx, ylab):
    b0 = c.canon(scene, "B0", "B100")
    b0p = c.canon(scene, "B0'", "B100")
    tmax = b0["cost"]["n_triangles"] / 1e6
    # anchor reference: horizontal line + diamond at full budget (neutral ink)
    y0 = b0["rendering_mean"]["psnr" if metric_idx == 1 else "lpips"]
    ax.axhline(y0, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.plot([tmax], [y0], marker="D", ms=5, color=INK2, zorder=3)
    if b0p is not None:
        yp = b0p["rendering_mean"]["psnr" if metric_idx == 1 else "lpips"]
        ax.axhline(yp, color=INK, lw=1.0, ls="solid", alpha=0.55, zorder=1)
        ax.plot([tmax], [yp], marker="D", ms=5, color=INK, zorder=3)
    for method, label, color in SERIES:
        pts = series_points(c, scene, method)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[metric_idx] for p in pts]
        ax.plot(xs, ys, "-o", color=color, lw=1.8, ms=5.5, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_title(scene, fontsize=9.5, color=INK)
    ax.set_xlim(0, tmax * 1.08)
    ax.tick_params(labelsize=8)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_xlabel("triangles [M]", fontsize=8.5)


def legend_handles(present):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=col, lw=1.8, marker="o", ms=5.5,
                 markeredgecolor=SURFACE, label=lab)
          for m, lab, col in SERIES if m in present]
    hs.append(Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 3)), marker="D",
                     ms=5, label="B0 clean@30k (legacy anchor)"))
    hs.append(Line2D([], [], color=INK, lw=1.0, alpha=0.55, marker="D", ms=5,
                     label="B0' clean-fixed@30k (primary anchor)"))
    return hs


def f2_pareto(c: Corpus, outdir):
    suites = {
        "srend": (["bicycle", "flowers", "garden", "stump", "treehill",
                   "room", "counter", "kitchen", "bonsai"], (3, 3)),
        "sgeo": (["ss3dm_town01", "ss3dm_town02", "ss3dm_town03",
                  "ss3dm_town06"], (2, 2)),
        "sdev": (["toy_parking", "toy_parking_v2", "toy_parking_occl",
                  "courtyard"], (2, 2)),
    }
    for tag, (scenes, (nr, nc)) in suites.items():
        for metric_idx, mname, ylab in ((1, "psnr", "PSNR [dB]"),
                                        (2, "lpips", "LPIPS")):
            fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.9 * nr + 0.7))
            axes = np.atleast_1d(axes).ravel()
            present = set()
            for ax, scene in zip(axes, scenes):
                draw_scene_panel(ax, c, scene, metric_idx, ylab)
                for m, _, _ in SERIES:
                    if series_points(c, scene, m):
                        present.add(m)
            for ax in axes[len(scenes):]:
                ax.axis("off")
            fig.suptitle(
                f"F2 — Pareto: {ylab} vs triangle count ({tag.upper()}); "
                "budgets B12.5/B25/B50; dual anchors", fontsize=11, color=INK)
            fig.legend(handles=legend_handles(present), loc="lower center",
                       ncol=3, frameon=False, fontsize=8.5)
            fig.tight_layout(rect=(0, 0.07 if nr > 1 else 0.16, 1, 0.96))
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(outdir, f"F2_pareto_{tag}_{mname}.{ext}"),
                            dpi=200)
            plt.close(fig)
            print(f"[figures] F2_pareto_{tag}_{mname}.png/.pdf")


def f1_teaser(c: Corpus, outdir):
    """DRAFT teaser: garden — budget vs quality vs geometry (humans polish)."""
    scene = "garden"
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    b0 = c.canon(scene, "B0", "B100")
    tmax = b0["cost"]["n_triangles"] / 1e6
    # (a) quality
    draw_scene_panel(axes[0], c, scene, 1, "PSNR [dB]")
    axes[0].set_title("quality: PSNR vs triangles", fontsize=9.5)
    # (b) perceptual
    draw_scene_panel(axes[1], c, scene, 2, "LPIPS")
    axes[1].set_title("perceptual: LPIPS vs triangles", fontsize=9.5)
    # (c) geometry: g3 floater fraction (single series; one axis)
    ax = axes[2]
    xs, ys = [], []
    for b in ("B12.5", "B25", "B50"):
        r = c.canon(scene, "B5", b)
        if r and r.get("g3") and "floater_triangle_fraction" in r["g3"]:
            xs.append(r["cost"]["n_triangles"] / 1e6)
            ys.append(r["g3"]["floater_triangle_fraction"] * 100)
    ax.plot(xs, ys, "-o", color=C_B5, lw=1.8, ms=5.5,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
    y0 = b0["g3"]["floater_triangle_fraction"] * 100
    ax.axhline(y0, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.plot([tmax], [y0], marker="D", ms=5, color=INK2)
    ax.set_xlim(0, tmax * 1.08)
    ax.set_xlabel("triangles [M]", fontsize=8.5)
    ax.set_ylabel("g3 floater tris [%]", fontsize=8.5)
    ax.set_title("geometry: floater fraction", fontsize=9.5)
    fig.suptitle("F1 (DRAFT) — garden: budget-quality-geometry at a glance "
                 "(B5 GEMS-core blue; diamonds/dashes = anchors)",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"F1_teaser_draft_garden.{ext}"), dpi=200)
    plt.close(fig)
    print("[figures] F1_teaser_draft_garden.png/.pdf")
    with open(os.path.join(outdir, "F1_caption.txt"), "w") as f:
        f.write(
            "F1 (draft caption): GEMS-core compaction on Mip-NeRF360 garden. "
            "Left/middle: rendering quality (PSNR, LPIPS) versus triangle "
            "count for evidence-guided pruning with drift-safe features-only "
            "fine-tuning (B5); at 50% triangles the model matches or exceeds "
            "both the deployed-default clean@30k anchor (dashed) and sits "
            "within the primary drift-repaired clean-fixed@30k anchor band "
            "(solid) — see T1/T2 for CIs. Right (honest caveat): the g3 "
            "floater-triangle fraction RISES after pruning on garden — "
            "importance pruning fragments connected components (E9 taxonomy "
            "family E; LEDGER GOAL#R-08 surprise (1)); GEMS does NOT claim "
            "geometry improvement (CLAIMS.md C2). Numbers: RESULTS/aggregate; "
            "per-view CIs in T1/T2.\n")


def f7_ablations(c: Corpus, outdir):
    rows = []  # (group, label, ci)

    def add(group, label, r, ref):
        if r is None or ref is None:
            return
        rows.append((group, label, paired_delta(r, ref, "psnr")))

    for s in ("garden", "toy_parking"):
        sn = "garden" if s == "garden" else "toy"
        ref = c.canon(s, "B4", "B50")
        add("FT channel (vs B4 no-FT)", f"default FT — {sn}",
            c.canon(s, "B5-ftdefault", "B50"), ref)
        add("FT channel (vs B4 no-FT)", f"lr x0.1 FT — {sn}",
            c.canon(s, "B5-ftlowlr", "B50"), ref)
        add("FT channel (vs B4 no-FT)", f"features-only FT (B5) — {sn}",
            c.canon(s, "B5", "B50"), ref)
        add("schedule (vs one-shot B5)", f"iterative 2-step — {sn}",
            c.canon(s, "B5-iter", "B50"), c.canon(s, "B5", "B50"))
    add("sourcing (vs 30k-sourced B5)", "26k-sourced — garden",
        c.canon("garden", "B5-src26k", "B50"), c.canon("garden", "B5", "B50"))
    e3 = [("e3_garden_B50_distill_v1", "e3_garden_B50_control_v1", "1x — garden"),
          ("e3_toy_B50_distill_v1", "e3_toy_B50_control_v1", "1x — toy"),
          ("e3v1_garden_B50_distill_v1", "e3_garden_B50_control_v1", "3x — garden"),
          ("e3v1_toy_parking_B50_distill_v1", "e3_toy_B50_control_v1", "3x — toy"),
          ("e3v2_garden_B50_distill_v1", "e3v2_garden_B50_control_v1", "3x+SH — garden"),
          ("e3v2_toy_B50_distill_v1", "e3v2_toy_B50_control_v1", "3x+SH — toy")]
    for d, ctrl, lab in e3:
        add("teacher distill - control (DEMOTED)", lab, c.by_dir(d), c.by_dir(ctrl))
    # importance vs random at the aggressive budget (all 9 S-REND scenes)
    for s in ("bicycle", "flowers", "garden", "stump", "treehill", "room",
              "counter", "kitchen", "bonsai"):
        add("evidence importance vs random (B12.5)", s,
            c.canon(s, "B5", "B12.5"), c.canon(s, "B2", "B12.5"))
    # importance-DEFINITION family (E6 sub-cell, GOAL#012 — axis flat)
    for s in ("garden", "kitchen", "ss3dm_town01"):
        ref = c.canon(s, "B5", "B50")
        add("importance definition (E6, flat axis)",
            f"max_blending vs pixels_total — {s}",
            c.canon(s, "B5-abl_blend", "B50"), ref)
        add("importance definition (E6, flat axis)",
            f"ckpt-stat vs pixels_total — {s}",
            c.canon(s, "B5-abl_ckptimp", "B50"), ref)
    # B3 QEM baseline vs B5 at matched budget (GOAL#013)
    for s in ("garden", "toy_parking", "courtyard"):
        add("B3 QEM+FT vs B5 (matched budget)", s,
            c.canon(s, "B3", "B50"), c.canon(s, "B5", "B50"))

    groups = []
    for g_, l_, ci in rows:
        if g_ not in groups:
            groups.append(g_)
    palette = (C_B5, C_B5I, C_B2, C_B6R, C_B4, C_B3, INK2)
    group_color = {g_: palette[i % len(palette)] for i, g_ in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(rows) + 1.8))
    y = 0
    yticks, ylabels = [], []
    for g_ in groups:
        grows = [r for r in rows if r[0] == g_]
        ax.text(0.0, y + 0.55, g_, transform=ax.get_yaxis_transform(),
                fontsize=8.5, fontweight="bold", color=INK)
        for _, label, ci in grows:
            col = group_color[g_]
            ax.plot([ci["ci_lo"], ci["ci_hi"]], [y, y], color=col, lw=1.8,
                    solid_capstyle="butt", zorder=3)
            ax.plot([ci["mean_diff"]], [y], "o", color=col, ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
            yticks.append(y)
            ylabels.append(label)
            y -= 1
        y -= 0.9
    ax.axvline(0, color=INK2, lw=1.0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("dPSNR vs reference [dB] (paired bootstrap 95% CI)")
    ax.set_title("F7 — Ablation deltas (E6 mapping of Stage-One variants)",
                 fontsize=10.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"F7_ablations.{ext}"), dpi=200)
    plt.close(fig)
    print("[figures] F7_ablations.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=ALL_ROWS_DEFAULT)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    with open(args.rows) as f:
        c = Corpus(json.load(f))
    os.makedirs(args.outdir, exist_ok=True)
    f2_pareto(c, args.outdir)
    f1_teaser(c, args.outdir)
    f7_ablations(c, args.outdir)


if __name__ == "__main__":
    main()
