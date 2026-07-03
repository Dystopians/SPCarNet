#!/usr/bin/env python
"""GEMS Stage-2 evidence pack — table generator (T1-T7, section 6 discipline).

Every number in every emitted table is computed HERE from
RESULTS/aggregate/all_rows.json (itself read from the metrics.json corpus),
the fps bench json, and the durable analysis/*.json artifacts — zero
hand-typed numbers. Paired deltas carry per-view bootstrap 95% CIs
(tools/gems/paired_bootstrap.py, 10k resamples, seed 0 — PROTOCOL section 5).

Outputs (markdown + csv) under RESULTS/aggregate/:
  T1_main_pareto      main Pareto summary per suite x budget (means +
                      per-scene win/iso/loss counts vs BOTH anchors)
  T2_rendering        per-scene rendering tables
  T3_geometry         g1-g4 per scene (VOID-aware; GT-CAL calibration row)
  T4_efficiency       tris/disk/VRAM/FPS@2res + pipeline overhead
  T5_downstream       d1/d2 per scene + R3 trilogy summary
  T6_ablations        Stage-One variant rows mapped to E6 columns
  T7_robustness       placeholder (PENDING: E7/E8 cells not yet run)

Usage:
    python tools/gems/report/tables.py [--rows FILE] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from tools.gems.paired_bootstrap import paired_bootstrap_ci  # noqa: E402

ALL_ROWS_DEFAULT = os.path.join(REPO_ROOT, "RESULTS", "aggregate", "all_rows.json")
OUTDIR_DEFAULT = os.path.join(REPO_ROOT, "RESULTS", "aggregate")
FPS_BENCH = os.path.join(REPO_ROOT, "RESULTS", "aggregate", "fps_bench_halfres.json")
ANALYSIS_ROOT = "/data/peilincai/gems_stage1/analysis"

# D3 floors (LEDGER STANDING CONSTRAINTS): rendering +0.10 dB / 0.005 LPIPS;
# compaction iso-quality floor: dPSNR >= -0.10 AND dLPIPS <= +0.005.
FLOOR_PSNR = 0.10
FLOOR_LPIPS = 0.005

SUITES = ("S-REND", "S-GEO", "S-DEV")
BUDGETS = ("B50", "B25", "B12.5")
SCENE_ORDER = ["bicycle", "flowers", "garden", "stump", "treehill", "room",
               "counter", "kitchen", "bonsai",
               "ss3dm_town01", "ss3dm_town02", "ss3dm_town03", "ss3dm_town06",
               "toy_parking", "courtyard"]

STATS_CAVEAT = (
    "STATS: every delta is a paired per-view bootstrap 95% CI (10k resamples, "
    "seed 0, PROTOCOL section 5). MULTIPLE-COMPARISONS CAVEAT (E10): dozens of "
    "CIs are reported across this pack; borderline CIs (effect near a floor or "
    "CI edge near 0) should be read with Bonferroni-style skepticism — "
    "headline claims rest only on effects that are large, replicated across "
    "scenes/suites, or mechanism-backed. Courtyard rendering CIs are 5-view "
    "(underpowered by design). Reporting language per section 6: 'improves/"
    "reduces' ONLY when the CI excludes 0 AND the D3 floor is cleared; "
    "otherwise 'comparable'/'inconclusive'."
)


# ---------------------------------------------------------------------------
# corpus access helpers
# ---------------------------------------------------------------------------

class Corpus:
    def __init__(self, payload):
        self.payload = payload
        self.rows = payload["rows"]

    def canon(self, scene, method, budget_label):
        """The unique canonical row for (scene, method, budget)."""
        hits = [r for r in self.rows
                if r["canonical"] and r["scene"] == scene
                and r["method"] == method and r["budget_label"] == budget_label]
        if len(hits) > 1:
            raise RuntimeError(
                f"non-unique canonical row {scene}/{method}/{budget_label}: "
                f"{[h['eval_dir'] for h in hits]}")
        return hits[0] if hits else None

    def by_dir(self, name):
        for r in self.rows:
            if r["eval_dir"] == name:
                return r
        return None

    def scenes(self, suite):
        return [s for s in SCENE_ORDER
                if any(r["scene"] == s and r["suite"] == suite for r in self.rows)]


def paired_delta(row_a, row_b, metric):
    """CI of mean(metric_a - metric_b), paired per test view (names asserted)."""
    pa, pb = row_a["per_view"], row_b["per_view"]
    if pa["image_names"] != pb["image_names"]:
        raise RuntimeError(f"view mismatch {row_a['eval_dir']} vs {row_b['eval_dir']}")
    return paired_bootstrap_ci(np.array(pa[metric]), np.array(pb[metric]))


def wil(ci, better="up"):
    """Classify a paired CI into win/iso/loss from the perspective of row_a."""
    lo, hi = ci["ci_lo"], ci["ci_hi"]
    if lo > 0:
        return "win" if better == "up" else "loss"
    if hi < 0:
        return "loss" if better == "up" else "win"
    return "iso"


def fmt_ci(ci, nd=3):
    return f"{ci['mean_diff']:+.{nd}f} [{ci['ci_lo']:+.{nd}f},{ci['ci_hi']:+.{nd}f}]"


def write_table(outdir, name, title, header_lines, columns, rows_out):
    """Emit <name>.md and <name>.csv."""
    md = io.StringIO()
    md.write(f"# {title}\n\n")
    stamp = datetime.now(timezone.utc).isoformat()
    md.write(f"_generated {stamp} by tools/gems/report/tables.py — every number "
             "computed from metrics.json-derived artifacts; none hand-typed._\n\n")
    for h in header_lines:
        md.write(f"> {h}\n")
    md.write("\n")
    md.write("| " + " | ".join(columns) + " |\n")
    md.write("|" + "|".join("---" for _ in columns) + "|\n")
    for row in rows_out:
        md.write("| " + " | ".join(str(c) for c in row) + " |\n")
    with open(os.path.join(outdir, f"{name}.md"), "w") as f:
        f.write(md.getvalue())
    with open(os.path.join(outdir, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows_out:
            w.writerow(row)
    print(f"[tables] wrote {name}.md / {name}.csv ({len(rows_out)} rows)")


# ---------------------------------------------------------------------------
# T1 — main Pareto summary
# ---------------------------------------------------------------------------

def t1_main_pareto(c: Corpus, outdir):
    cols = ["suite", "budget", "method", "scenes", "tri_ratio_vs_B0",
            "mean dPSNR vs B0 [dB]", "PSNR w/i/l vs B0",
            "mean dLPIPS vs B0", "LPIPS w/i/l vs B0",
            "mean dPSNR vs B0'", "PSNR w/i/l vs B0'",
            "mean dLPIPS vs B0'", "LPIPS w/i/l vs B0'",
            "iso-floor pass (vs B0)", "iso-floor pass (vs B0')",
            "mean FPS x vs B0"]
    out = []
    detail = []  # per-scene records for the csv companion / F2
    for suite in SUITES:
        scenes = c.scenes(suite)
        for budget in BUDGETS:
            # B7 (teacher) is EXCLUDED here: three e3 variants exist per scene
            # (no unique row; demoted diagnostic) — all appear in T6 instead.
            for method in ("B5", "B4", "B2", "B5-iter", "B6R"):
                recs = []
                for s in scenes:
                    r = c.canon(s, method, budget)
                    b0 = c.canon(s, "B0", "B100")
                    if r is None or b0 is None:
                        continue
                    b0p = c.canon(s, "B0'", "B100")
                    d = {"suite": suite, "scene": s, "method": method,
                         "budget": budget,
                         "tri_ratio": r["cost"]["n_triangles"] / b0["cost"]["n_triangles"],
                         "fps_ratio": r["cost"]["render_fps"] / b0["cost"]["render_fps"],
                         "psnr": r["rendering_mean"]["psnr"],
                         "lpips": r["rendering_mean"]["lpips"],
                         "dpsnr_b0": paired_delta(r, b0, "psnr"),
                         "dlpips_b0": paired_delta(r, b0, "lpips")}
                    if b0p is not None:
                        d["dpsnr_b0p"] = paired_delta(r, b0p, "psnr")
                        d["dlpips_b0p"] = paired_delta(r, b0p, "lpips")
                    recs.append(d)
                if not recs:
                    continue
                def counts(key, better):
                    ws = [wil(d[key], better) for d in recs if key in d]
                    return (f"{sum(w=='win' for w in ws)}/"
                            f"{sum(w=='iso' for w in ws)}/"
                            f"{sum(w=='loss' for w in ws)}") if ws else "—"
                iso_pass = sum(
                    d["dpsnr_b0"]["mean_diff"] >= -FLOOR_PSNR
                    and d["dlpips_b0"]["mean_diff"] <= FLOOR_LPIPS for d in recs)
                with_b0p = [d for d in recs if "dpsnr_b0p" in d]
                iso_pass_p = (
                    f"{sum(d['dpsnr_b0p']['mean_diff'] >= -FLOOR_PSNR and d['dlpips_b0p']['mean_diff'] <= FLOOR_LPIPS for d in with_b0p)}"
                    f"/{len(with_b0p)}") if with_b0p else "—"
                def mean_of(key):
                    vals = [d[key]["mean_diff"] for d in recs if key in d]
                    return f"{np.mean(vals):+.3f}" if vals else "—"
                out.append([
                    suite, budget, method, f"{len(recs)}/{len(scenes)}",
                    f"{np.mean([d['tri_ratio'] for d in recs]):.3f}",
                    mean_of("dpsnr_b0"), counts("dpsnr_b0", "up"),
                    mean_of("dlpips_b0"), counts("dlpips_b0", "down"),
                    mean_of("dpsnr_b0p"), counts("dpsnr_b0p", "up"),
                    mean_of("dlpips_b0p"), counts("dlpips_b0p", "down"),
                    f"{iso_pass}/{len(recs)}", iso_pass_p,
                    f"{np.mean([d['fps_ratio'] for d in recs]):.2f}",
                ])
                detail.extend(recs)
    header = [
        STATS_CAVEAT,
        "Anchors: B0 = clean@30k (legacy/deployed default); B0' = clean-fixed@30k "
        "(PRIMARY anchor per LEDGER GOAL#R-01; exists on S-REND only — S-GEO/S-DEV "
        "anchor columns vs B0' are blank by construction).",
        "w/i/l = per-scene win/iso/loss counts by paired 95% CI (win = CI "
        "excludes 0 in the improving direction). iso-floor pass = scenes with "
        f"mean dPSNR >= -{FLOOR_PSNR} AND mean dLPIPS <= +{FLOOR_LPIPS} vs B0 "
        "(D3 compaction iso-quality floor).",
        "MISSING BY DESIGN (honesty, section 7.3/7.4): B1 no-op, B3 QEM+FT and "
        "H1/R1 reference rows have NOT been run (MATRIX: TODO); B2 exists at "
        "B12.5 on all 9 S-REND scenes but only on garden/toy/courtyard at "
        "B50/B25 (e1b era). B6/B7 appear only as diagnostic rows on dev scenes "
        "(both DEMOTED per CLAIMS.md). S-GEO B2 was never run.",
    ]
    write_table(outdir, "T1_main_pareto", "T1 — Main Pareto summary "
                "(E1-PARETO / E10)", header, cols, out)

    # per-scene companion csv (feeds F2 and the spot-check)
    with open(os.path.join(outdir, "T1_per_scene_detail.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "scene", "method", "budget", "tri_ratio", "psnr",
                    "lpips", "dpsnr_b0", "dpsnr_b0_lo", "dpsnr_b0_hi",
                    "dlpips_b0", "dpsnr_b0p", "dpsnr_b0p_lo", "dpsnr_b0p_hi",
                    "dlpips_b0p", "fps_ratio"])
        for d in detail:
            w.writerow([d["suite"], d["scene"], d["method"], d["budget"],
                        f"{d['tri_ratio']:.4f}", f"{d['psnr']:.4f}",
                        f"{d['lpips']:.4f}",
                        f"{d['dpsnr_b0']['mean_diff']:.4f}",
                        f"{d['dpsnr_b0']['ci_lo']:.4f}",
                        f"{d['dpsnr_b0']['ci_hi']:.4f}",
                        f"{d['dlpips_b0']['mean_diff']:.5f}",
                        *( [f"{d['dpsnr_b0p']['mean_diff']:.4f}",
                            f"{d['dpsnr_b0p']['ci_lo']:.4f}",
                            f"{d['dpsnr_b0p']['ci_hi']:.4f}",
                            f"{d['dlpips_b0p']['mean_diff']:.5f}"]
                           if "dpsnr_b0p" in d else ["", "", "", ""]),
                        f"{d['fps_ratio']:.3f}"])
    print("[tables] wrote T1_per_scene_detail.csv")
    return detail


# ---------------------------------------------------------------------------
# T2 — per-scene rendering
# ---------------------------------------------------------------------------

def t2_rendering(c: Corpus, outdir):
    cols = ["suite", "scene", "method", "budget", "PSNR", "SSIM", "LPIPS",
            "dPSNR vs B0 (CI)", "dPSNR vs B0' (CI)", "dLPIPS vs B0 (CI)",
            "n views", "eval row"]
    out = []
    methods = [("B0", "B100"), ("B0'", "B100"), ("B0-26k", "B100")] + [
        (m, b) for b in BUDGETS for m in ("B5", "B4", "B2", "B5-iter", "B6R")]
    for suite in SUITES:
        for s in c.scenes(suite):
            b0 = c.canon(s, "B0", "B100")
            b0p = c.canon(s, "B0'", "B100")
            for method, budget in methods:
                r = c.canon(s, method, budget)
                if r is None:
                    continue
                dp = fmt_ci(paired_delta(r, b0, "psnr")) if b0 and r is not b0 else "—"
                dpp = fmt_ci(paired_delta(r, b0p, "psnr")) if b0p and r is not b0p else "—"
                dl = fmt_ci(paired_delta(r, b0, "lpips"), nd=4) if b0 and r is not b0 else "—"
                out.append([suite, s, method, budget,
                            f"{r['rendering_mean']['psnr']:.3f}",
                            f"{r['rendering_mean']['ssim']:.4f}",
                            f"{r['rendering_mean']['lpips']:.4f}",
                            dp, dpp, dl,
                            r["cost"]["n_test_views"], r["eval_dir"]])
    header = [
        STATS_CAVEAT,
        "B0-26k rows are the 26k-snapshot context anchor (NOT compute-matched; "
        "LEDGER GOAL#R-01). Teacher-headroom analysis (E3-REND): see "
        "T6_ablations teacher block — distill-control deltas were REAL but "
        "sub-floor (E3 sunset; CLAIMS.md C3').",
    ]
    write_table(outdir, "T2_rendering", "T2 — Per-scene rendering (E3-REND)",
                header, cols, out)


# ---------------------------------------------------------------------------
# T3 — geometry reliability
# ---------------------------------------------------------------------------

def g(r, fam, key, nd=4, scale=1.0):
    v = r.get(fam)
    if v is None:
        return "—"
    if "VOID" in v:
        return "VOID"
    if "skipped" in v:
        return "n/a"
    if key not in v:
        return "—"
    return f"{v[key]*scale:.{nd}f}" if isinstance(v[key], float) else str(v[key])


def t3_geometry(c: Corpus, outdir):
    cols = ["suite", "scene", "method", "budget", "g1 free-space viol.",
            "g2 depth L1 [m]", "g3 floater comps", "g3 floater frac",
            "g4 chamfer [m]", "g4 F@5cm", "eval row"]
    out = []
    methods = [("GT-CAL", "B100"), ("B0", "B100")] + [
        (m, b) for b in BUDGETS for m in ("B5", "B4", "B6R")]
    for suite in SUITES:
        for s in c.scenes(suite):
            for method, budget in methods:
                r = c.canon(s, method, budget)
                if r is None:
                    continue
                out.append([suite, s, method, budget,
                            g(r, "g1", "value"), g(r, "g2", "value", 4),
                            g(r, "g3", "floater_component_count"),
                            g(r, "g3", "floater_triangle_fraction", 5),
                            g(r, "g4", "chamfer_l1_m"), g(r, "g4", "fscore_at_tau"),
                            r["eval_dir"]])
    header = [
        "GT-CAL = toy GT-mesh model calibration row (metric-suite validation, "
        "LEDGER GOAL#004). VOID cells are LEDGER-voided measurements (encoded "
        "in collect.py), NOT missing data: pre-R-08 SS3DM g4 used a raw-cm "
        "unmirrored GT mesh. 'n/a' = metric not defined for the scene (no GT "
        "asset / no gt_depth).",
        "SS3DM g1 absolute values are NOT comparable across datasets (GT depth "
        "to ~1000 m; LEDGER GOAL#R-08). g4 absolutes on SS3DM are "
        "GT-sampling-density-limited; paired deltas are the valid signal.",
        "C2 is a DEMOTED measurement claim (CLAIMS.md): these tables document "
        "preservation + the photometric-masks-geometry finding, not improvement.",
    ]
    write_table(outdir, "T3_geometry", "T3 — Geometry reliability (E2-GEO)",
                header, cols, out)


# ---------------------------------------------------------------------------
# T4 — efficiency
# ---------------------------------------------------------------------------

def t4_efficiency(c: Corpus, outdir):
    bench = {}
    bench_meta = "PENDING (fps_bench_halfres.json absent)"
    if os.path.exists(FPS_BENCH):
        with open(FPS_BENCH) as f:
            bj = json.load(f)
        bench = {b["eval_dir"]: b for b in bj["results"] if "render_fps_halfres" in b}
        bench_meta = (f"{bj['label']}; loop: {bj['fps_loop']}; GPU {bj['gpu_name']}")
    jobs = c.payload.get("job_wallclocks_min", {})

    cols = ["suite", "scene", "method", "budget", "triangles", "disk MB",
            "peak VRAM MB", "FPS @protocol res", "FPS @0.5x res (bench-only)",
            "prune+FT overhead [min]", "overhead vs 30k-train", "eval row"]
    out = []
    methods = [("B0", "B100")] + [(m, b) for b in BUDGETS for m in ("B5", "B4")]
    for suite in SUITES:
        for s in c.scenes(suite):
            for method, budget in methods:
                r = c.canon(s, method, budget)
                if r is None:
                    continue
                bkey = r["eval_dir"]
                bfps = f"{bench[bkey]['render_fps_halfres']:.1f}" if bkey in bench else "—"
                sw = r["provenance"].get("stage_wallclock_sec")
                if not sw:
                    # canonical re-eval rows (geo_v1 / *_v4) have no row.json;
                    # their stage stamps live on the superseded pipeline twin
                    # of the SAME checkpoint (duplicate-verified in collect.py)
                    for twin in c.rows:
                        if (twin["scene"] == s and twin["method"] == method
                                and twin["budget_label"] == budget
                                and twin["provenance"].get("stage_wallclock_sec")):
                            sw = twin["provenance"]["stage_wallclock_sec"]
                            break
                if sw:
                    overhead_min = (sw.get("evidence", 0) + sw.get("prune", 0)
                                    + sw.get("finetune", 0)) / 60.0
                    ov = f"{overhead_min:.1f}"
                    # measured 30k-train reference exists for SS3DM towns
                    jname = f"b0_{s}" if s.startswith("ss3dm") else None
                    if jname and jname in jobs:
                        ratio = overhead_min / jobs[jname]["wallclock_min"]
                        ovr = f"{ratio*100:.0f}% of measured {jobs[jname]['wallclock_min']:.0f} min"
                    else:
                        ovr = "see note (no measured 30k train for this scene)"
                else:
                    ov, ovr = "—", "—"
                out.append([suite, s, method, budget,
                            r["cost"]["n_triangles"],
                            f"{r['cost']['disk_mb']:.0f}",
                            f"{r['cost']['peak_vram_mb']:.0f}",
                            f"{r['cost']['render_fps']:.1f}", bfps,
                            ov, ovr, r["eval_dir"]])
    # measured training wallclocks appendix (from supervised job files)
    train_rows = [[k, f"{v['wallclock_min']:.1f}", v["exit_code"]]
                  for k, v in sorted(jobs.items()) if k.startswith("b0_")]
    header = [
        f"Second-resolution FPS column: {bench_meta}. These half-res numbers "
        "are for the E4 efficiency table only ('bench-only, non-protocol "
        "resolution') — no quality metric was computed at this resolution.",
        "HONESTY CAVEAT on FPS columns: the protocol-res FPS in each "
        "metrics.json was measured at eval time (idle GPUs, various same-model "
        "RTX 6000 Ada devices); the half-res bench ran on GPU 4 WITH a "
        "background process from another user (~21% util at launch). "
        "Method-vs-method comparisons WITHIN the half-res column share GPU "
        "state (paired-ish); protocol-vs-half-res comparisons across columns "
        "are indicative only. Several large scenes show half-res FPS close to "
        "or below protocol-res FPS — consistent with triangle-sort-bound "
        "rendering (cost dominated by primitive count, not pixels) plus "
        "contention; do not read that cross-column delta as a resolution "
        "scaling law.",
        "Pipeline overhead = evidence+prune+finetune wall-clock from row.json "
        "stage stamps (measured). 30k-train reference: MEASURED for SS3DM "
        "towns (supervised job files, appendix below). For M360/toy/courtyard "
        "the original 30k trainings predate the job supervisor: LEDGER "
        "GOAL#002 estimates 40-80 min/scene (M360, images_4/2) and GOAL#004 "
        "measured ~17 min (toy); those are ESTIMATES, labeled as such — the "
        "overhead column stays measured either way.",
        "Overhead rows exist only for pipeline rows with row.json (B5/B4/B2 "
        "tags s2/e1b/e1v2...); B4 overhead is evidence+prune only (no FT).",
        "Measured 30k-train wall-clocks (supervised jobs): "
        + "; ".join(f"{r[0]}={r[1]} min (exit {r[2]})" for r in train_rows),
    ]
    write_table(outdir, "T4_efficiency", "T4 — Efficiency (E4-EFF)", header,
                cols, out)


# ---------------------------------------------------------------------------
# T5 — downstream
# ---------------------------------------------------------------------------

def t5_downstream(c: Corpus, outdir):
    cols = ["suite", "scene", "method", "budget", "d1 false-free rate",
            "d1 false-occupied rate", "d2 agreement", "d2 unsafe disagreement",
            "n traj", "eval row"]
    out = []
    methods = [("GT-CAL", "B100"), ("B0", "B100")] + [
        (m, b) for b in BUDGETS for m in ("B5", "B4", "B6R")]
    for suite in SUITES:
        for s in c.scenes(suite):
            for method, budget in methods:
                r = c.canon(s, method, budget)
                if r is None or r.get("d1") is None:
                    continue
                if "skipped" in (r.get("d1") or {}):
                    continue
                out.append([suite, s, method, budget,
                            g(r, "d1", "false_free_rate"),
                            g(r, "d1", "false_occupied_rate"),
                            g(r, "d2", "agreement_rate"),
                            g(r, "d2", "unsafe_disagreement_rate"),
                            g(r, "d2", "n_traj"), r["eval_dir"]])
    header = [
        "d1 = occupancy confusion at 0.10 m voxels (false-free is the "
        "safety-critical direction); d2 = collision-verdict agreement on 200 "
        "seed-0 trajectories. C4' is a PRESERVATION claim (CLAIMS.md): no row "
        "here claims improvement vs clean.",
        "R3 consumption trilogy summary below is read by script from "
        "analysis/r3{a,c,b}_*/summary.json (LEDGER #R-02/#R-03/#R-06).",
    ]
    write_table(outdir, "T5_downstream", "T5 — Downstream proxies (E5-DOWN)",
                header, cols, out)

    # ---- R3 trilogy appendix (script-read from durable analysis artifacts) ----
    lines = ["# T5b — R3 consumption-trilogy summary (script-extracted)\n"]
    r3a = json.load(open(os.path.join(ANALYSIS_ROOT, "r3a_occupancy_routes", "summary.json")))
    lines.append("\n## R3.a occupancy routes (GOAL#R-02)\n\nVerdict (verbatim "
                 f"summary.json): `{json.dumps(r3a.get('verdict'))[:800]}`\n")
    lines.append("\n| cell | route-i false-free | route-ii false-free | "
                 "route-i false-occ | route-ii false-occ |\n|---|---|---|---|---|\n")
    models = r3a.get("models", {})
    for cell, v in models.items():
        ri = v.get("route_i", {}).get("d1", v.get("route_i", {}))
        rii = v.get("route_ii", {}).get("d1", v.get("route_ii", {}))
        def _g(d, k):
            return f"{d[k]:.4f}" if isinstance(d, dict) and k in d else "?"
        lines.append(f"| {cell} | {_g(ri,'false_free_rate')} | "
                     f"{_g(rii,'false_free_rate')} | {_g(ri,'false_occupied_rate')} | "
                     f"{_g(rii,'false_occupied_rate')} |\n")
    r3c = json.load(open(os.path.join(ANALYSIS_ROOT, "r3c_planner", "summary.json")))
    lines.append("\n## R3.c planner loop v0 (GOAL#R-03)\n\nVerdict (verbatim "
                 f"summary.json): `{json.dumps(r3c.get('verdict'))[:800]}`\n")
    lines.append("\n| cell | plans found /100 | collisions /100 plans | "
                 "path inflation vs GTREF |\n|---|---|---|---|\n")
    def _f(x, fmt="{:.1f}"):
        return fmt.format(x) if isinstance(x, (int, float)) else "n/a"
    for cell, v in (r3c.get("cells") or {}).items():
        lines.append(f"| {cell} | {v.get('plans_found','?')} | "
                     f"{_f(v.get('collisions_per_100_plans'))} | "
                     f"{_f(v.get('path_length_inflation_vs_gtref'), '{:+.3f}')} |\n")
    r3b = json.load(open(os.path.join(ANALYSIS_ROOT, "r3b_submesh", "summary.json")))
    lines.append("\n## R3.b certified sub-mesh (GOAL#R-06)\n\nVerdict (verbatim "
                 f"summary.json): `{json.dumps(r3b.get('verdict'))[:800]}`\n")
    lines.append("\n| cell | found /100 | coll /100 | d1 ff (sub-mesh) | "
                 "d1 ff (raw route-i) | kept frac of finite |\n|---|---|---|---|---|---|\n")
    for cell, v in (r3b.get("cells") or {}).items():
        pl = v.get("planner", {})
        lines.append(f"| {cell} | {pl.get('plans_found','?')} | "
                     f"{_f(pl.get('collisions_per_100_plans'))} | "
                     f"{v['d1_submesh']['false_free_rate']:.4f} | "
                     f"{v['d1_raw_route_i']['false_free_rate']:.4f} | "
                     f"{v['kept']['kept_fraction_of_finite']:.4f} |\n")
    with open(os.path.join(outdir, "T5b_r3_trilogy.md"), "w") as f:
        f.writelines(lines)
    print("[tables] wrote T5b_r3_trilogy.md")


# ---------------------------------------------------------------------------
# T6 — ablations (existing Stage-One variant rows mapped to E6 columns)
# ---------------------------------------------------------------------------

def t6_ablations(c: Corpus, outdir):
    cols = ["ablation axis", "variant", "scene", "budget",
            "PSNR", "dPSNR vs reference (CI)", "dLPIPS vs reference (CI)",
            "reference row", "eval row"]
    out = []

    def add(axis, variant, r, ref):
        if r is None or ref is None:
            return
        out.append([axis, variant, r["scene"], r["budget_label"],
                    f"{r['rendering_mean']['psnr']:.3f}",
                    fmt_ci(paired_delta(r, ref, "psnr")),
                    fmt_ci(paired_delta(r, ref, "lpips"), nd=4),
                    ref["eval_dir"], r["eval_dir"]])

    # (a) FT channel / loss-form: default FT vs lr x0.1 vs features-only, vs B4
    for s in ("garden", "toy_parking"):
        for b in ("B50", "B25"):
            ref = c.canon(s, "B4", b)
            for meth, name in (("B5-ftdefault", "default all-param FT (e1b)"),
                               ("B5-ftlowlr", "all-param FT, lr x0.1 (e1v1)"),
                               ("B5", "features-only FT (e1v2/s2) = GEMS-core")):
                add("FT channel (E6 loss-form)", name, c.canon(s, meth, b), ref)
    # (b) prune schedule: one-shot vs iterative (vs same-scene B5 one-shot)
    for s in ("garden", "toy_parking"):
        add("prune schedule", "iterative 2-step (e1v3) vs one-shot",
            c.canon(s, "B5-iter", "B50"), c.canon(s, "B5", "B50"))
    # (c) importance family: evidence vs random (matched FT config)
    for s in SCENE_ORDER:
        for b in BUDGETS:
            add("importance family", "evidence importance vs random (both +FT)",
                c.canon(s, "B5", b), c.canon(s, "B2", b))
    # (d) sourcing probe
    add("checkpoint sourcing", "26k-sourced prune+FT vs 30k-sourced (e26src)",
        c.canon("garden", "B5-src26k", "B50"), c.canon("garden", "B5", "B50"))
    # (e) teacher distillation variants: distill vs its OWN control
    e3_pairs = [
        ("e3_garden_B50_distill_v1", "e3_garden_B50_control_v1", "density 1x"),
        ("e3_toy_B50_distill_v1", "e3_toy_B50_control_v1", "density 1x"),
        ("e3v1_garden_B50_distill_v1", "e3_garden_B50_control_v1",
         "density 3x (control: e3 same-length)"),
        ("e3v1_toy_parking_B50_distill_v1", "e3_toy_B50_control_v1",
         "density 3x (control: e3 same-length)"),
        ("e3v2_garden_B50_distill_v1", "e3v2_garden_B50_control_v1",
         "density 3x + SH rest-lr 1.0"),
        ("e3v2_toy_B50_distill_v1", "e3v2_toy_B50_control_v1",
         "density 3x + SH rest-lr 1.0"),
    ]
    for dname, cname, variant in e3_pairs:
        add("teacher distillation (E3, DEMOTED diagnostic)",
            variant, c.by_dir(dname), c.by_dir(cname))
    # (f) geometry mechanisms vs their B5 baseline (rendering guard view)
    for s in ("garden", "toy_parking", "courtyard"):
        b5 = c.canon(s, "B5", "B50")
        for meth, name in (("B6-losses", "free-space+depth losses (m3/E2 a1)"),
                           ("B6-gradrouted", "gradient-routed losses (m3v1/E2 v1)"),
                           ("B6-floaterprune", "evidence floater deletion (e2v3)"),
                           ("B6R", "opacity release + fade-prune (e2r)")):
            add("geometry mechanism (E2 family, all FAIL/bounded)",
                name, c.canon(s, meth, "B50"), b5)

    header = [
        STATS_CAVEAT,
        "E6 mapping note: these are the EXISTING Stage-One/1R variant rows "
        "mapped onto the E6 ablation axes (MATRIX E6: 'map them in'). "
        "Un-run E6 axes remain open: importance-feature families beyond "
        "evidence-vs-random, and reallocation on/off (never built).",
        "Geometry-mechanism rows are diagnostics of FAILED/demoted mechanisms "
        "(E2/E2R/E3 verdicts in LEDGER; NEGATIVE_RESULTS.md) — shown with the "
        "rendering guard delta; their g-metric movements are in T3/LEDGER.",
    ]
    write_table(outdir, "T6_ablations", "T6 — Ablations (E6-ABL)", header,
                cols, out)


# ---------------------------------------------------------------------------
# T7 — robustness placeholder
# ---------------------------------------------------------------------------

def t7_robustness(outdir):
    with open(os.path.join(outdir, "T7_robustness.md"), "w") as f:
        f.write("# T7 — Robustness / sensitivity (E7-SENS, E8-ROBUST)\n\n"
                "**STATUS: PENDING.** No E7 (seeds / dev-vs-full resolution / "
                "loss-weight 3-point) or E8 (view drop / pose noise / S-GEN) "
                "cells have been run yet (MATRIX: TODO, Tier 2). This file is "
                "a placeholder so the evidence pack is honest about the gap; "
                "it will be regenerated by tables.py once those cells exist.\n")
    print("[tables] wrote T7_robustness.md (placeholder PENDING)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=ALL_ROWS_DEFAULT)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    with open(args.rows) as f:
        c = Corpus(json.load(f))
    os.makedirs(args.outdir, exist_ok=True)
    t1_main_pareto(c, args.outdir)
    t2_rendering(c, args.outdir)
    t3_geometry(c, args.outdir)
    t4_efficiency(c, args.outdir)
    t5_downstream(c, args.outdir)
    t6_ablations(c, args.outdir)
    t7_robustness(args.outdir)


if __name__ == "__main__":
    main()
