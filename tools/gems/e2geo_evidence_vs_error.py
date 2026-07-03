#!/usr/bin/env python
"""GEMS Stage-2 E2-GEO evidence-vs-error ANALYSIS (LEDGER GOAL #015 PART A).

ANALYSIS, NOT METHOD (Stage-2 prompt section 5 E2 + section 7.5): does
TRAIN-view per-triangle evidence predict TEST-time per-triangle rendering
error?  This is where the v3xx evidence/certificate machinery appears in the
paper -- explicitly framed as analysis.  Nothing computed here feeds back into
any model, prune, or training run.

Per scene (B0 clean checkpoint):
  1. TRAIN evidence = the cached gems_pipeline evidence npz (D4 train-only,
     fingerprint-matched; produced by tools/gems/triangle_evidence.py).
     Columns used: pixels_total, views_seen,
     residual_view_mean = residual_sum / residual_pixels, max_blending_max.
  2. TEST error = render every TEST view via tools/gems/eval_context
     (training-time settings, supersampling x4); per-pixel |render - GT| L1
     (mean over RGB, native res, float -- the SAME convention as
     triangle_evidence's train residual, deliberately NOT the 8-bit metric
     path, so train and test residuals are scale-comparable); attribute to
     triangles via rend_ids with the g3-verified gate
     (ids >= 0) & (ids < T) & (depth_full > 0) at supersampled resolution,
     residual nearest-upsampled (byte-for-byte the triangle_evidence.py
     convention).  Per-triangle TEST error = err_sum / err_pixels.
  3. Statistics: Spearman rank correlations (test error vs each evidence
     column) over triangles owning >= 1 test pixel; equal-count decile
     reliability curves (stable-argsort deciles); selection-effect tail table
     binned by train views_seen.

Pre-registered predictions (LEDGER GOAL #015, frozen before any number):
  P-A1: rho(residual_view_mean, test_err) in [+0.2, +0.8] on all 3 scenes.
  P-A2: (i) mean(test_err - residual_view_mean) larger for views_seen <= 5
        than for views_seen > 30 on every scene; (ii) pixels_total and
        views_seen rank-correlate NEGATIVELY with test error, and the lowest
        pixels_total decile has higher mean test error than the top decile.

Usage:
    python tools/gems/e2geo_evidence_vs_error.py --scene garden \
        --checkpoint <pt> --evidence-npz <npz> --out-root <dir> [--gpu N]
    python tools/gems/e2geo_evidence_vs_error.py --aggregate --out-root <dir>

Durable: /data/peilincai/gems_stage1/analysis/e2geo_evidence_vs_error/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Same guard as triangle_evidence / geometry_metrics: rend_ids is float32.
MAX_TRIANGLES_FLOAT32_IDS = 2 ** 24

EVIDENCE_COLUMNS = ("pixels_total", "views_seen", "residual_view_mean",
                    "max_blending_max")
VIEWS_SEEN_BINS = ((0, 0), (1, 1), (2, 5), (6, 10), (11, 30), (31, 10 ** 9))
N_DECILES = 10

OUT_ROOT_DEFAULT = "/data/peilincai/gems_stage1/analysis/e2geo_evidence_vs_error"

# Okabe-Ito (colorblind-safe; same convention as planner_loop panels).
C_LINE = "#0072B2"
C_MASS = "#E69F00"


# ---------------------------------------------------------------------------
# TEST-view per-triangle error accumulation (GPU)
# ---------------------------------------------------------------------------

def accumulate_test_error(ctx, n_triangles, log_every=10):
    """Render all TEST views; accumulate per-triangle |render-GT| L1 via
    rend_ids with the exact triangle_evidence / g3 gating conventions."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = "cuda"
    err_sum = torch.zeros(n_triangles, dtype=torch.float64, device=device)
    err_pixels = torch.zeros(n_triangles, dtype=torch.int64, device=device)
    test_views_seen = torch.zeros(n_triangles, dtype=torch.int32, device=device)

    test_cams = ctx.test_cams
    assert len(test_cams) > 0, "scene has no test cameras"
    t0 = time.time()
    view_names = []
    for i, cam in enumerate(test_cams):
        view_names.append(cam.image_name)
        pkg = ctx.render_view(cam)

        ids = pkg["rend_ids"].detach().reshape(-1)
        depth_full = pkg.get("depth_full")
        valid = (ids >= 0) & (ids < n_triangles)
        if depth_full is not None:
            valid &= depth_full.detach().reshape(-1) > 0
        idx = ids[valid].round().long()

        render = pkg["render"].detach().clamp(0.0, 1.0)
        gt = cam.original_image[:3].to(render.device).clamp(0.0, 1.0)
        resid_native = (render - gt).abs().mean(dim=0)
        hs, ws = pkg["depth_full"].shape[-2:] if depth_full is not None \
            else pkg["rend_ids"].shape[-2:]
        resid_up = F.interpolate(
            resid_native[None, None], size=(int(hs), int(ws)), mode="nearest"
        ).reshape(-1)

        counts = torch.bincount(idx, minlength=n_triangles)
        err_sum += torch.bincount(
            idx, weights=resid_up[valid].double(), minlength=n_triangles)
        err_pixels += counts
        test_views_seen += (counts > 0).to(torch.int32)

        del pkg, ids, valid, idx, counts, resid_native, resid_up
        if (i + 1) % log_every == 0 or (i + 1) == len(test_cams):
            print(f"[e2geo] test views {i + 1}/{len(test_cams)} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        torch.cuda.empty_cache()

    return (err_sum.cpu().numpy(), err_pixels.cpu().numpy(),
            test_views_seen.cpu().numpy(), view_names,
            time.time() - t0)


# ---------------------------------------------------------------------------
# Statistics (CPU, numpy/scipy)
# ---------------------------------------------------------------------------

def stable_deciles(values):
    """Deterministic equal-count decile assignment: stable argsort (ties
    broken by triangle id), decile = rank * 10 // n.  Returns int array."""
    import numpy as np
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return (rank * N_DECILES) // n


def decile_table(evidence, test_err, err_sum):
    import numpy as np
    dec = stable_deciles(evidence)
    total_mass = float(err_sum.sum())
    rows = []
    for d in range(N_DECILES):
        m = dec == d
        rows.append({
            "decile": d + 1,
            "n": int(m.sum()),
            "evidence_mean": float(evidence[m].mean()),
            "evidence_min": float(evidence[m].min()),
            "evidence_max": float(evidence[m].max()),
            "test_err_mean": float(test_err[m].mean()),
            "test_err_median": float(np.median(test_err[m])),
            "err_mass_share": (float(err_sum[m].sum()) / total_mass
                               if total_mass > 0 else None),
        })
    return rows


def views_seen_tail_table(views_seen, test_err, resid_mean, covered_pixels_total):
    """Selection-effect tail: per train-coverage bin, the gap between test
    error and the train-residual prediction."""
    import numpy as np
    rows = []
    for lo, hi in VIEWS_SEEN_BINS:
        m = (views_seen >= lo) & (views_seen <= hi)
        row = {"views_seen_bin": f"{lo}" if lo == hi else
               (f"{lo}-{hi}" if hi < 10 ** 9 else f">={lo}"),
               "n": int(m.sum())}
        if row["n"] == 0:
            rows.append(row)
            continue
        row["test_err_mean"] = float(test_err[m].mean())
        # train residual is defined only where the triangle was train-visible
        md = m & (covered_pixels_total > 0)
        row["n_train_visible"] = int(md.sum())
        if md.sum() > 0:
            row["train_resid_mean"] = float(resid_mean[md].mean())
            row["gap_mean_test_minus_train"] = float(
                (test_err[md] - resid_mean[md]).mean())
            row["ratio_test_over_train"] = (
                float(test_err[md].mean() / resid_mean[md].mean())
                if resid_mean[md].mean() > 0 else None)
        rows.append(row)
    return rows


def analyze_scene(scene, checkpoint, evidence_npz, out_root, gpu=None):
    import numpy as np

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from scipy.stats import spearmanr
    from tools.gems.scenes import SCENES
    from tools.gems.eval_context import build_eval_context
    from tools.gems.triangle_evidence import checkpoint_fingerprint

    spec = SCENES[scene]
    out_dir = os.path.join(out_root)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    # --- load TRAIN evidence (cached, fingerprint-verified) ---
    ev = np.load(evidence_npz)
    meta = json.loads(bytes(ev["meta_json"]).decode())
    fp = checkpoint_fingerprint(checkpoint)
    assert meta["checkpoint"]["sha256_first16mb"] == fp["sha256_first16mb"], (
        f"evidence npz fingerprint mismatch: {evidence_npz} was computed for "
        f"{meta['checkpoint']['path']}, not {checkpoint}")
    pixels_total = ev["pixels_total"].astype(np.int64)
    views_seen = ev["views_seen"].astype(np.int64)
    residual_sum = ev["residual_sum"].astype(np.float64)
    residual_pixels = ev["residual_pixels"].astype(np.int64)
    max_blending_max = ev["max_blending_max"].astype(np.float64)
    n_triangles = pixels_total.shape[0]
    assert n_triangles < MAX_TRIANGLES_FLOAT32_IDS
    resid_mean = np.divide(residual_sum, residual_pixels,
                           out=np.zeros_like(residual_sum),
                           where=residual_pixels > 0)

    # --- TEST-view error pass (GPU) ---
    print(f"[e2geo] {scene}: building eval context ...", flush=True)
    ctx = build_eval_context(checkpoint, spec)
    faces_n = int(ctx.faces().shape[0])
    assert faces_n == n_triangles, (
        f"checkpoint has {faces_n} triangles but evidence npz has {n_triangles}")
    err_sum, err_pixels, test_views_seen, test_view_names, render_sec = \
        accumulate_test_error(ctx, n_triangles)
    # D4-orientation note: test GT is consumed HERE, on the eval side only,
    # exactly as run_eval consumes it; the output is an analysis artifact.

    npz_path = os.path.join(out_dir, f"{scene}_test_error.npz")
    np.savez_compressed(
        npz_path, err_sum=err_sum, err_pixels=err_pixels,
        test_views_seen=test_views_seen,
        test_view_names=np.asarray(test_view_names))

    # --- per-triangle populations ---
    covered = err_pixels > 0                       # owns >= 1 test pixel
    test_err = np.divide(err_sum, err_pixels,
                         out=np.zeros_like(err_sum), where=covered)
    grp_gap = covered & (pixels_total == 0)        # never train-visible
    result = {
        "scene": scene,
        "checkpoint": fp,
        "evidence_npz": os.path.abspath(evidence_npz),
        "n_triangles": int(n_triangles),
        "n_test_views": len(test_view_names),
        "n_train_views_in_evidence": meta["n_views_used"],
        "render_wallclock_sec": render_sec,
        "population": {
            "n_test_covered": int(covered.sum()),
            "frac_test_covered": float(covered.mean()),
            "n_test_covered_never_train_visible": int(grp_gap.sum()),
            "coverage_gap_group": {
                "n": int(grp_gap.sum()),
                "test_err_mean": (float(test_err[grp_gap].mean())
                                  if grp_gap.any() else None),
                "err_mass_share": (float(err_sum[grp_gap].sum() /
                                         err_sum.sum())
                                   if err_sum.sum() > 0 else None),
                "note": ("test-covered triangles with pixels_total == 0: no "
                         "train residual defined; excluded ONLY from the "
                         "residual_view_mean correlation, reported here"),
            },
        },
        "spearman": {}, "deciles": {}, "tail": None, "predictions": {},
    }

    cols = {
        "pixels_total": pixels_total.astype(np.float64),
        "views_seen": views_seen.astype(np.float64),
        "residual_view_mean": resid_mean,
        "max_blending_max": max_blending_max,
    }
    for name, col in cols.items():
        m = covered.copy()
        note = None
        if name == "residual_view_mean":
            m &= pixels_total > 0
            note = "train-visible & test-covered triangles only"
        rho, pval = spearmanr(col[m], test_err[m])
        result["spearman"][name] = {
            "rho": float(rho), "p": float(pval), "n": int(m.sum()),
            **({"note": note} if note else {}),
        }
        result["deciles"][name] = decile_table(col[m], test_err[m], err_sum[m])
        print(f"[e2geo] {scene}: spearman({name}) = {rho:+.4f} "
              f"(n={int(m.sum()):,})", flush=True)

    result["tail"] = views_seen_tail_table(
        views_seen[covered], test_err[covered], resid_mean[covered],
        pixels_total[covered])

    # --- pre-registered prediction checks (per scene) ---
    rho_res = result["spearman"]["residual_view_mean"]["rho"]
    tail = {r["views_seen_bin"]: r for r in result["tail"]}

    def _gap(bins):
        import numpy as np
        num = sum(t["gap_mean_test_minus_train"] * t["n_train_visible"]
                  for b in bins if (t := tail.get(b)) and
                  t.get("n_train_visible", 0) > 0)
        den = sum(t["n_train_visible"] for b in bins
                  if (t := tail.get(b)) and t.get("n_train_visible", 0) > 0)
        return num / den if den else None

    gap_low = _gap(["1", "2-5"])          # views_seen <= 5, train-visible
    gap_high = _gap([">=31"])
    dec_pt = result["deciles"]["pixels_total"]
    result["predictions"] = {
        "P-A1_rho_residual_in_band": bool(0.2 <= rho_res <= 0.8),
        "P-A1_rho_residual": rho_res,
        "P-A2i_gap_low_gt_high": (bool(gap_low > gap_high)
                                  if None not in (gap_low, gap_high) else None),
        "P-A2i_gap_views_seen_le5": gap_low,
        "P-A2i_gap_views_seen_gt30": gap_high,
        "P-A2ii_rho_pixels_total_neg": bool(
            result["spearman"]["pixels_total"]["rho"] < 0),
        "P-A2ii_rho_views_seen_neg": bool(
            result["spearman"]["views_seen"]["rho"] < 0),
        "P-A2ii_bottom_decile_gt_top": bool(
            dec_pt[0]["test_err_mean"] > dec_pt[-1]["test_err_mean"]),
    }

    with open(os.path.join(out_dir, f"{scene}_result.json"), "w") as f:
        json.dump(result, f, indent=1)
    make_plots(scene, result, os.path.join(out_dir, "plots"))
    print(f"[e2geo] {scene}: wrote {scene}_result.json + plots", flush=True)
    return result


# ---------------------------------------------------------------------------
# Reliability plots (matplotlib, static analysis artifact)
# ---------------------------------------------------------------------------

def make_plots(scene, result, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, name in zip(axes, EVIDENCE_COLUMNS):
        rows = result["deciles"][name]
        x = [r["decile"] for r in rows]
        y = [r["test_err_mean"] for r in rows]
        mass = [r["err_mass_share"] for r in rows]
        ax.plot(x, y, "-o", color=C_LINE, lw=2, ms=5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.bar(x, [m * max(y) / max(max(mass), 1e-12) for m in mass],
               color=C_MASS, alpha=0.25, width=0.7, zorder=1)
        ax.set_title(f"{name}\n(rho={result['spearman'][name]['rho']:+.3f})",
                     fontsize=9)
        ax.set_xlabel("train-evidence decile (1=lowest)", fontsize=8)
        ax.set_xticks(x)
        ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("mean per-pixel test |render-GT| L1", fontsize=8)
    fig.suptitle(
        f"{scene} (B0 clean): TEST error vs TRAIN evidence deciles "
        f"(line = mean test error; orange bars = test-error mass share, "
        f"rescaled)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(plots_dir, f"{scene}_reliability.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # tail plot: gap (test - train residual) per views_seen bin
    rows = [r for r in result["tail"] if "gap_mean_test_minus_train" in r]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    xs = range(len(rows))
    ax.bar(xs, [r["gap_mean_test_minus_train"] for r in rows],
           color=C_LINE, width=0.62, zorder=3)
    ax.axhline(0, color="#666666", lw=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["views_seen_bin"] for r in rows], fontsize=8)
    ax.set_xlabel("train views_seen bin", fontsize=9)
    ax.set_ylabel("mean(test err - train resid)", fontsize=9)
    ax.set_title(f"{scene}: evidence under-estimation gap by train coverage",
                 fontsize=10)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{scene}_coverage_gap.png"), dpi=150)
    plt.close(fig)


def aggregate(out_root, scenes):
    results = {}
    for s in scenes:
        with open(os.path.join(out_root, f"{s}_result.json")) as f:
            results[s] = json.load(f)
    verdict = {
        "P-A1_all_scenes": all(
            r["predictions"]["P-A1_rho_residual_in_band"] for r in results.values()),
        "P-A2i_all_scenes": all(
            bool(r["predictions"]["P-A2i_gap_low_gt_high"]) for r in results.values()),
        "P-A2ii_all_scenes": all(
            r["predictions"]["P-A2ii_rho_pixels_total_neg"] and
            r["predictions"]["P-A2ii_rho_views_seen_neg"] and
            r["predictions"]["P-A2ii_bottom_decile_gt_top"]
            for r in results.values()),
    }
    summary = {
        "goal": "LEDGER GOAL #015 PART A (E2-GEO evidence-vs-error analysis)",
        "framing": ("ANALYSIS, not method: correlation of train-view "
                    "evidence/certificate columns with actual per-pixel "
                    "test error (Stage-2 prompt section 5 E2, section 7.5)"),
        "scenes": {s: {
            "spearman": {k: v["rho"] for k, v in r["spearman"].items()},
            "population": r["population"],
            "tail": r["tail"],
            "predictions": r["predictions"],
        } for s, r in results.items()},
        "verdict": verdict,
    }
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(verdict, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene")
    ap.add_argument("--checkpoint")
    ap.add_argument("--evidence-npz")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--agg-scenes", default="garden,kitchen,ss3dm_town01")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.out_root, args.agg_scenes.split(","))
        return
    assert args.scene and args.checkpoint and args.evidence_npz
    analyze_scene(args.scene, args.checkpoint, args.evidence_npz,
                  args.out_root, gpu=args.gpu)


if __name__ == "__main__":
    main()
