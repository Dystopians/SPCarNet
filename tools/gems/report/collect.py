#!/usr/bin/env python
"""GEMS Stage-2 evidence pack — corpus collector (Stage2 Prompt section 8).

Walks the eval corpus at /data/peilincai/gems_stage1/eval/*/metrics.json into
RESULTS/aggregate/all_rows.json: one flat record per eval row with

  * scene / suite, method tag parsed into the B-zoo taxonomy, budget,
  * per-view rendering arrays (psnr/ssim/lpips + image names),
  * cost scalars (tris/disk/VRAM/FPS),
  * geometry (g1..g4) and downstream (d1/d2) scalars,
  * provenance (checkpoint sha, git commit, config hash, timestamps,
    stage wall-clocks from row.json),
  * role (anchor/primary/ablation/diagnostic/...), canonical flag, and
    VOID annotations for LEDGER-voided metric fields.

VOID list (encoded from LEDGER.md — do not weaken):
  * courtyard_clean30k_v2 g4       (GOAL#005 CRASH/VOID#2, scan alignment,
                                    commit cb1560b; superseded by v3/v4)
  * every pre-#R-08 SS3DM g4 field (GOAL#R-08: raw-cm unmirrored GT mesh ->
                                    garbage chamfer). Pack v3 (GOAL#020): the
                                    discriminator is now read from the
                                    artifact itself — a pre-R-08 g4 lacks the
                                    'roi_presample_crop' marker in
                                    g4.gt_source that the fixed mesh path
                                    stamps. This keys the VOID on the
                                    measurement provenance instead of the
                                    dirname heuristic (which wrongly VOIDed
                                    post-fix rows such as b6r_* and the
                                    GOAL#019 gap rows).

No numbers are typed here: everything is read from metrics.json / row.json.

NOT corpus rows (by design): the R1 3DGS cross-representation reference lives
OUTSIDE the single mouth (sanctioned exception, LEDGER GOAL#017) at
analysis/r1_3dgs_reference/ — tables.py quotes it as clearly-marked CONTEXT
lines in T1/T4, never as corpus rows. Same for the H1 v106 historical context
row (analysis/h1_v106_context/, GOAL#013).

Usage:
    python tools/gems/report/collect.py [--eval-root DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EVAL_ROOT_DEFAULT = "/data/peilincai/gems_stage1/eval"
OUT_DEFAULT = os.path.join(REPO_ROOT, "RESULTS", "aggregate", "all_rows.json")
ANALYSIS_ROOT = "/data/peilincai/gems_stage1/analysis"
JOBS_ROOT = "/data/peilincai/gems_stage1/jobs"

M360_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill",
               "room", "counter", "kitchen", "bonsai"}
SS3DM_SCENES = {"ss3dm_town01", "ss3dm_town02", "ss3dm_town03", "ss3dm_town06"}
# D-2 toy variants (GOAL#016) are S-DEV family members of toy_parking
DEV_SCENES = {"toy_parking", "courtyard", "toy_parking_v2", "toy_parking_occl"}


def suite_of(scene: str) -> str:
    if scene in M360_SCENES:
        return "S-REND"
    if scene in SS3DM_SCENES:
        return "S-GEO"
    if scene in DEV_SCENES:
        return "S-DEV"
    return "OTHER"


# ---------------------------------------------------------------------------
# B-zoo taxonomy mapping (task/MATRIX.md):
#   B0   clean30k anchor              B0'  cleanfixed30k primary anchor
#   B0-26k clean26k context anchor    B1   no-op pass-through sanity (b1noop)
#   B2   random prune + safe FT       B2-noft random prune, no FT (context)
#   B4   evidence prune, no FT        B5   GEMS-core evidence prune + feat-FT
#   B5-iter iterative schedule (e1v3) B6R  opacity-release (e2r)
#   B5-seed1 E7 seed-sensitivity pair (GOAL#019)
#   B6-* geometry-loss diagnostics (m3/m3v1/e2v3)
#   B7-diag teacher distill (e3*)     B7-control same-length no-teacher FT
#   GT-CAL toy GT-mesh calibration model
# Budget labels: B50/B25/B12.5/B6.25 (row.json budget_label(0.0625)='B6',
#   normalized here to B6.25) and B100 for full budget.
# Roles: anchor | primary | context | ablation | diagnostic | intermediate |
#        superseded | calibration | audit | sanity
# ---------------------------------------------------------------------------

VOID_NOTES = {
    "courtyard_clean30k_v2": {
        "g4": "VOID per LEDGER GOAL#005 CRASH/VOID#2: ETH3D scan_alignment "
              "transforms missing in this row (fixed commit cb1560b; "
              "superseded by courtyard_clean30k_v3/v4)."
    },
}
SS3DM_PRE_R08_G4_VOID = ("VOID per LEDGER GOAL#R-08: pre-R-08 SS3DM g4 used the "
                         "raw-cm, unmirrored GT mesh (chamfer 11k-77k m garbage); "
                         "valid g4 = rows whose g4.gt_source carries the R-08 "
                         "'roi_presample_crop' marker (the *_geo_v1 re-evals and "
                         "every eval run after the fix).")

# Rows that are duplicate evals of the SAME checkpoint through the same mouth.
# canonical=the most complete / current-protocol row; the others get
# role='superseded' plus a consistency check that rendering means agree.
DUPLICATE_SETS = [
    # (canonical, [superseded...])
    ("courtyard_clean30k_v4", ["courtyard_clean30k_v1", "courtyard_clean30k_v2",
                               "courtyard_clean30k_v3"]),
    ("courtyard_B50_ft_v4", ["courtyard_B50_importance_ft_e1v2",
                             "courtyard_B50_importance_ft_e1v2_dsv1"]),
    ("courtyard_B50_importance_noft_v4", ["courtyard_B50_importance_noft_e1b"]),
    ("courtyard_B25_importance_noft_v4", ["courtyard_B25_importance_noft_e1b"]),
    ("ss3dm_town01_clean30k_geo_v1", ["ss3dm_town01_clean30k_v1"]),
    ("ss3dm_town02_clean30k_geo_v1", ["ss3dm_town02_clean30k_v1"]),
    ("ss3dm_town03_clean30k_geo_v1", ["ss3dm_town03_clean30k_v1"]),
    ("ss3dm_town06_clean30k_geo_v1", ["ss3dm_town06_clean30k_v1"]),
    ("ss3dm_town01_B50_geo_v1", ["ss3dm_town01_B50_importance_ft_s2"]),
    ("ss3dm_town02_B50_geo_v1", ["ss3dm_town02_B50_importance_ft_s2"]),
    ("ss3dm_town03_B50_geo_v1", ["ss3dm_town03_B50_importance_ft_s2"]),
    ("ss3dm_town06_B50_geo_v1", ["ss3dm_town06_B50_importance_ft_s2"]),
]

# Special-name rows (no row.json, or name doesn't follow <scene>_<Bnn>_<mode>_<tag>)
SPECIAL = {
    "toy_parking_GTmodel_v1": ("GT-CAL", "calibration", 1.0, "B100"),
    "toy_cleanresume_diag": ("B0+FT-diag", "diagnostic", 1.0, "B100"),
    "garden_clean26k_diag": ("B0-26k", "diagnostic", 1.0, "B100"),
    "garden_audit_v2": (None, "audit", None, None),
    "toy_parking_audit": (None, "audit", None, None),
    "stage2_entry_audit": (None, "audit", None, None),
    "stage2_entry_audit2": (None, "audit", None, None),
    "e2r_audit_courtyard": (None, "audit", None, None),
    "b3_audit_garden": (None, "audit", None, None),
    "d2_audit_v2_B5": (None, "audit", None, None),
}

_BUDGET_RE = re.compile(r"_B(\d+)_")


def _budget_from_name(dirname: str):
    m = _BUDGET_RE.search(dirname)
    if not m:
        return None, None
    n = int(m.group(1))
    # B12 rows are budget=0.125 / B6 rows budget=0.0625 in row.json; labels
    # normalized to B12.5 / B6.25 (run-pipeline budget_label truncates).
    label = {12: "B12.5", 6: "B6.25"}.get(n, f"B{n}")
    value = {12: 0.125, 6: 0.0625}.get(n, n / 100.0)
    return value, label


def classify(dirname: str, row: dict | None, scene: str):
    """Return (method, role, budget_value, budget_label, variant_note)."""
    if dirname in SPECIAL:
        method, role, bv, bl = SPECIAL[dirname]
        return method, role, bv, bl, ""

    # --- Stage3 closure rows (GOAL#C-01; T7 consumes these explicitly) ---
    if dirname == "garden_clean30k_seed1_v1":
        return "B0-seed1", "robustness", 1.0, "B100", (
            "Stage3 W1 seed-1 clean full retrain substitute row")
    if dirname == "garden_halftrain_clean30k_v1":
        return "B0-halftrain", "robustness", 1.0, "B100", (
            "Stage3 W2 garden 50% train-view-drop clean retrain row")

    # --- anchors / clean rows (no row.json) ---
    if "cleanfixed30k" in dirname:
        return "B0'", "anchor", 1.0, "B100", "primary anchor (R1, features-only 26k->30k)"
    if "clean30k" in dirname:
        return "B0", "anchor", 1.0, "B100", "legacy/deployed-default anchor"
    if "clean26k" in dirname:
        return "B0-26k", "context", 1.0, "B100", "26k-snapshot context anchor (not compute-matched)"

    # --- GOAL#019 gap-closure rows ---
    if dirname == "garden_B50_seed1_v1":
        # E7 seed-sensitivity pair: identical prune + features-only FT chain
        # re-run end-to-end with seed 1 (checkpoint models/garden_B50_seed1);
        # the paired delta vs the canonical seed-0 B5@B50 row is computed by
        # script in T7 (tables.py) — never typed here.
        return "B5-seed1", "ablation", 0.5, "B50", (
            "GOAL#019 E7 seed-sensitivity pair (seed 1 re-run of the "
            "garden B5@B50 chain; paired vs seed-0 row in T7)")

    # --- named experiment families (no row.json) ---
    if dirname.startswith("e2r_"):
        bv, bl = _budget_from_name(dirname)
        return "B6R", "ablation", bv, bl, "E2R opacity-floor release + fade-prune (GOAL#R-04)"
    if dirname.startswith("b6r_"):
        # B6R-on-S-GEO cells: owning goal CLOSED (LEDGER GOAL#014,
        # DONE-FAIL as pre-registered — g3 FRACTION arm 0/3 towns; PSNR
        # guard held 3/3; LPIPS/g1/g3-components better CI-excl-0 3/3;
        # d1-ff worse CI-excl-0 3/3). Folded into tables 2026-07-03.
        bv, bl = _budget_from_name(dirname)
        return "B6R", "ablation", bv, bl, (
            "B6R-on-SS3DM (GOAL#014 DONE-FAIL as pre-registered; B6R stays "
            "a bounded courtyard-scoped positive, NOT claim-grade)")
    if dirname.startswith("e2v3_"):
        bv, bl = _budget_from_name(dirname)
        role = "diagnostic" if dirname.endswith("_diag") else "ablation"
        note = "E2v3 evidence-based floater deletion (GOAL#008)"
        if dirname.endswith("_diag"):
            note += " — pruned-no-FT diagnostic"
        return "B6-floaterprune", role, bv, bl, note
    if dirname.startswith("m3v1_"):
        bv, bl = _budget_from_name(dirname)
        return "B6-gradrouted", "ablation", bv, bl, "E2 variant 1 gradient-routed geometry losses (GOAL#007)"
    if dirname.startswith("m3_e2_"):
        bv, bl = _budget_from_name(dirname)
        return "B6-losses", "ablation", bv, bl, "E2 attempt 1 free-space+depth-consistency losses (GOAL#006)"
    if dirname.startswith(("e3_", "e3v1_", "e3v2_")):
        bv, bl = _budget_from_name(dirname)
        kind = "distill" if "_distill_" in dirname else "control"
        gen = dirname.split("_")[0]  # e3 | e3v1 | e3v2
        variant = {"e3": "density 1x", "e3v1": "density 3x",
                   "e3v2": "density 3x + SH rest-lr 1.0"}[gen]
        method = "B7-diag" if kind == "distill" else "B7-control"
        return method, "ablation", bv, bl, f"E3 teacher {kind} ({variant}, GOAL#007)"

    # --- rows with row.json (pipeline rows) ---
    if row is not None:
        mode = row.get("mode", "")
        tag = row.get("tag", "")
        bv = row.get("budget")
        bl = row.get("budget_label")
        if bl == "B12":
            bl = "B12.5"
        elif bl == "B6":
            # budget_label(0.0625) truncates to 'B6' in row.json — normalize
            # to B6.25 (GOAL#019 far-end budget rows; avoids clashing with
            # the B6 method name).
            bl = "B6.25"
        note = f"tag={tag}"
        if tag == "b1noop":
            # B1 no-op pass-through sanity (GOAL#019 gap: MATRIX E1 B1 cell,
            # run instead of waived): budget=1.0, importance_noft — the
            # pipeline must reproduce the clean checkpoint exactly. The
            # exactness check (paired delta vs B0 identically zero, same
            # triangle count) is computed by script in T2/T7.
            return "B1", "sanity", bv, bl, (
                note + " (B1 no-op pass-through sanity, GOAL#019)")
        if isinstance(tag, str) and tag.startswith("abl_"):
            # E6 importance-family ablation rows: owning goal CLOSED (LEDGER
            # GOAL#012 — revision trigger NOT tripped; pixels_total stands;
            # family axis flat, all pairwise |dPSNR| <= 0.052 dB; measured
            # pipeline noise floor 1.6e-5 dB on the town01 degenerate row).
            fam = {"abl_blend": "max_blending_max",
                   "abl_ckptimp": "ckpt_importance_score"}.get(tag, tag)
            return f"B5-{tag}", "ablation", bv, bl, (
                note + f" (E6 importance family = {fam}, GOAL#012 closed)")
        if mode == "qem_ft":
            # B3 QEM-decimation + safe-FT column (LEDGER GOAL#013 DONE-PASS)
            return "B3", "primary", bv, bl, (
                note + " (QEM decimation via fast_simplification + "
                "features-only FT; GOAL#013)")
        if tag == "e1":
            # first e1 chain, pre supersampling-scaling fix -> superseded by e1b
            return f"pre-fix {mode}", "superseded", bv, bl, note + " (pre-scaling-fix, superseded by e1b)"
        if tag in ("diag1k", "reprocheck") or dirname.endswith("_diag1k"):
            return "B5-diag", "diagnostic", bv, bl, note
        if "reprocheck" in dirname:
            return "B5", "audit", bv, bl, note + " (fresh-clone reproduction check)"
        if mode == "random_ft":
            return "B2", "primary", bv, bl, note
        if mode == "random_noft":
            return "B2-noft", "context", bv, bl, note
        if mode == "importance_noft":
            return "B4", "primary", bv, bl, note
        if mode == "importance_ft":
            if tag == "stage3seed1":
                return "B5-seed1-full", "robustness", bv, bl, (
                    note + " (Stage3 W1 B5@B50 from seed-1 clean retrain)")
            if tag == "stage3drop50":
                return "B5-halftrain", "robustness", bv, bl, (
                    note + " (Stage3 W2 B5@B50 under 50% train-view drop)")
            if tag in ("s2", "e1v2"):
                return "B5", "primary", bv, bl, note
            if tag == "e1b":
                return "B5-ftdefault", "ablation", bv, bl, note + " (default all-param FT — drift destroyer)"
            if tag == "e1v1":
                return "B5-ftlowlr", "ablation", bv, bl, note + " (lr x0.1 FT variant)"
            if tag == "e26src":
                return "B5-src26k", "ablation", bv, bl, note + " (26k-sourced probe)"
            if tag == "e1v3":
                # final model of the 2-step iterative schedule; budget in
                # row.json is relative to the step-1 model (0.5/0.71 = 0.704),
                # i.e. exactly 50% of the CLEAN model — normalize to B50.
                return "B5-iter", "ablation", 0.5, "B50", (
                    note + f" (iterative schedule FINAL; row.json budget={bv} "
                    "is relative to step-1; = 50% of clean, tri-count verified)")
            if tag == "e1v3s1":
                return "B5-iter-step1", "intermediate", bv, bl, note + " (iterative schedule step 1)"
        return f"{mode}", "diagnostic", bv, bl, note + " (unmapped tag)"

    # --- residual no-row.json rows: courtyard v4 / dsv1 re-evals ---
    if dirname.endswith("_v4") or dirname.endswith("_dsv1"):
        bv, bl = _budget_from_name(dirname)
        if "noft" in dirname:
            return "B4", "primary", bv, bl, "downstream-enabled re-eval (same ckpt as e1b row)"
        return "B5", "primary", bv, bl, "downstream-enabled re-eval (same ckpt as e1v2 row)"
    if dirname.endswith("_geo_v1"):
        bv, bl = _budget_from_name(dirname + "_")
        if "clean30k" in dirname:
            return "B0", "anchor", 1.0, "B100", "R-08 geometry re-eval (valid g1/g2/g4/d1/d2)"
        return "B5", "primary", bv or 0.5, bl or "B50", "R-08 geometry re-eval (valid g1/g2/g4/d1/d2)"

    return "UNKNOWN", "diagnostic", None, None, "unclassified"


def _num(d: dict, *keys):
    out = {}
    for k in keys:
        if k in d:
            out[k] = d[k]
    return out


def extract(dirname: str, eval_root: str):
    mdir = os.path.join(eval_root, dirname)
    mpath = os.path.join(mdir, "metrics.json")
    with open(mpath) as f:
        m = json.load(f)
    row = None
    rpath = os.path.join(mdir, "row.json")
    if os.path.exists(rpath):
        with open(rpath) as f:
            row = json.load(f)

    scene = m.get("scene", "?")
    method, role, budget, budget_label, note = classify(dirname, row, scene)

    geometry = m.get("geometry", {}) if isinstance(m.get("geometry"), dict) else {}
    downstream = m.get("downstream", {}) if isinstance(m.get("downstream"), dict) else {}

    def fam(d, name, fields):
        sub = d.get(name)
        if not isinstance(sub, dict):
            return None
        if "skipped" in sub:
            return {"skipped": sub["skipped"]}
        return _num(sub, *fields)

    rec = {
        "eval_dir": dirname,
        "eval_path": mdir,
        "scene": scene,
        "suite": suite_of(scene),
        "method": method,
        "role": role,
        "budget": budget,
        "budget_label": budget_label,
        "note": note,
        "rendering_mean": m["rendering"]["mean"],
        "per_view": m["rendering"]["per_view"],
        "cost": _num(m.get("cost", {}), "n_triangles", "n_vertices", "disk_mb",
                     "peak_vram_mb", "render_fps", "n_test_views"),
        "g1": fam(geometry, "g1", ["value", "n_samples", "n_violations", "source"]),
        "g2": fam(geometry, "g2", ["value", "n_views"]),
        "g3": fam(geometry, "g3", ["floater_component_count",
                                   "floater_triangle_fraction", "n_components",
                                   "n_triangles", "n_support_views"]),
        "g4": fam(geometry, "g4", ["chamfer_l1_m", "fscore_at_tau",
                                   "precision_at_tau", "recall_at_tau", "tau_m",
                                   "n_gt_samples", "gt_source"]),
        "d1": fam(downstream, "d1", ["false_free_rate", "false_occupied_rate",
                                     "n_gt_occupied", "n_gt_free", "n_voxels",
                                     "voxel_m"]),
        "d2": fam(downstream, "d2", ["agreement_rate", "unsafe_disagreement_rate",
                                     "n_traj", "n_gt_collision",
                                     "n_recon_collision", "seed"]),
        "void": {},
        "provenance": {
            "protocol_version": m.get("protocol_version"),
            "eval_git_commit": m.get("git_commit"),
            "eval_timestamp_utc": m.get("timestamp_utc"),
            "checkpoint": m.get("checkpoint"),
            "metrics_json": mpath,
        },
        "canonical": True,  # may be flipped below by duplicate resolution
    }
    if row is not None:
        rec["provenance"].update({
            "config_hash": row.get("config_hash"),
            "pipeline_git_commit": row.get("git_commit"),
            "source_ckpt": row.get("source_ckpt"),
            "ft_iters": row.get("ft_iters"),
            "ft_wallclock_min": row.get("ft_wallclock_min"),
            "stage_wallclock_sec": row.get("stage_wallclock_sec"),
            "n_triangles_clean": row.get("n_triangles_clean"),
            "n_triangles_pruned": row.get("n_triangles_pruned"),
            "wandb_name": row.get("wandb_name"),
        })

    # ---- VOID annotations (LEDGER-encoded) ----
    if dirname in VOID_NOTES:
        rec["void"].update(VOID_NOTES[dirname])
    if scene in SS3DM_SCENES:
        # Pre-R-08 SS3DM g4 is VOID (raw-cm unmirrored GT mesh). Keyed on the
        # artifact: the R-08 fixed mesh path stamps '|roi_presample_crop'
        # into g4.gt_source; anything without the marker predates the fix.
        # (Pack v3 refinement over the old dirname-endswith-_geo_v1 rule,
        # which wrongly VOIDed post-fix rows: b6r_*, GOAL#019 gap rows.)
        g4v = rec["g4"] or {}
        if (g4v and "skipped" not in g4v
                and "roi_presample_crop" not in str(g4v.get("gt_source", ""))):
            rec["void"]["g4"] = SS3DM_PRE_R08_G4_VOID
    for famname, why in rec["void"].items():
        if isinstance(rec.get(famname), dict):
            rec[famname] = {"VOID": why}
    return rec


def job_wallclocks():
    """Wall-clock of supervised jobs from pidfile->exitfile mtimes (measured,
    durable; used by T4 for the SS3DM full-training reference)."""
    out = {}
    if not os.path.isdir(JOBS_ROOT):
        return out
    for f in sorted(os.listdir(JOBS_ROOT)):
        if not f.endswith(".pid"):
            continue
        name = f[:-4]
        pidf = os.path.join(JOBS_ROOT, f)
        exitf = os.path.join(JOBS_ROOT, name + ".exit")
        if os.path.exists(exitf):
            with open(exitf) as fh:
                code = fh.read().strip()
            out[name] = {
                "wallclock_min": (os.path.getmtime(exitf) - os.path.getmtime(pidf)) / 60.0,
                "exit_code": code,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=EVAL_ROOT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--link-tree", action="store_true",
                    help="materialize the section-8 RESULTS/<suite>/<scene>/"
                         "<method>/<budget> tree as symlinks to the canonical "
                         "eval rows (no data duplication)")
    args = ap.parse_args()

    dirs = sorted(d for d in os.listdir(args.eval_root)
                  if os.path.isfile(os.path.join(args.eval_root, d, "metrics.json")))
    rows = [extract(d, args.eval_root) for d in dirs]
    by_dir = {r["eval_dir"]: r for r in rows}

    # ---- duplicate resolution + consistency checks ----
    consistency = []
    for canon, dupes in DUPLICATE_SETS:
        if canon not in by_dir:
            continue
        c = by_dir[canon]
        for d in dupes:
            if d not in by_dir:
                continue
            r = by_dir[d]
            same = (abs(c["rendering_mean"]["psnr"] - r["rendering_mean"]["psnr"]) < 1e-6
                    and c["cost"].get("n_triangles") == r["cost"].get("n_triangles"))
            consistency.append({"canonical": canon, "duplicate": d,
                                "rendering_identical": bool(same)})
            r["canonical"] = False
            r["superseded_by"] = canon
            if not same:
                r["note"] += " !! RENDERING MISMATCH vs canonical duplicate — investigate"
    # non-table roles are never canonical for aggregation
    for r in rows:
        if r["role"] in ("audit", "superseded", "diagnostic", "intermediate",
                         "ablation-pending"):
            r["canonical"] = False

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        head = f"unavailable ({exc})"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": os.path.abspath(__file__),
        "repo_head_at_collect": head,
        "eval_root": args.eval_root,
        "n_rows": len(rows),
        "void_policy": {
            "courtyard_clean30k_v2.g4": VOID_NOTES["courtyard_clean30k_v2"]["g4"],
            "ss3dm_pre_R08_g4": SS3DM_PRE_R08_G4_VOID,
        },
        "duplicate_consistency": consistency,
        "job_wallclocks_min": job_wallclocks(),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    if args.link_tree:
        base = os.path.dirname(os.path.dirname(args.out))  # RESULTS/
        n_links = 0
        for r in rows:
            if not r["canonical"]:
                continue
            d = os.path.join(base, r["suite"], r["scene"], r["method"],
                             r["budget_label"] or "NA")
            os.makedirs(d, exist_ok=True)
            link = os.path.join(d, "eval_row")
            if os.path.islink(link):
                os.unlink(link)
            os.symlink(r["eval_path"], link)
            n_links += 1
        print(f"[collect] linked section-8 tree: {n_links} canonical rows under {base}/<suite>/")

    n_canon = sum(1 for r in rows if r["canonical"])
    n_bad = sum(1 for c in consistency if not c["rendering_identical"])
    print(f"[collect] {len(rows)} rows ({n_canon} canonical) -> {args.out}")
    print(f"[collect] duplicate consistency: {len(consistency)} checked, "
          f"{n_bad} mismatches")
    if n_bad:
        for c in consistency:
            if not c["rendering_identical"]:
                print("  MISMATCH:", c)
        sys.exit(1)


if __name__ == "__main__":
    main()
