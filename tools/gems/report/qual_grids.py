"""GEMS Stage 2 — E11-QUAL qualitative grids (MATRIX cell E11-QUAL).

Per-suite qualitative grids over the banked single-mouth eval rows:
  rows   = {clean, B50 B5, B25 B5} (+ B6R row where a b6r/e2r eval row exists)
  cols   = 3 view-blocks x {RGB render, median-depth map, floater overlay}
  blocks = the Stage-2 prompt SS5 anti-cherry-picking crop rule, NO exceptions:
    (a) BEST-case crop      view = argmax banked per-view PSNR of the B50 B5 row
    (b) MEDIAN-PSNR view    view = image_names[argsort(psnr)[(n-1)//2]] of the
                            SAME banked array -- chosen BY THIS SCRIPT, printed
    (c) FAILURE crop        view = the E9-taxonomy view where a curated case
                            names one; else argmin banked per-view PSNR

Crop windows (frozen rule, GOAL #014): computed ONCE from the B50 B5 row's
|render-GT| error map at the selected view (best view -> argmin-error
half-frame window; failure view -> argmax-error half-frame window; median
view -> full frame), then applied IDENTICALLY to every model row.

Floater overlays re-use each row's OWN banked geometry/g3_floaters.npz ids
(no recomputation). Depth colormap vmax = P99 of the CLEAN row's valid depth
at that view, shared down the column (turbo, the PROTOCOL panels.py cmap).

Renders are regenerated at the banked checkpoints recorded in each eval row's
metrics.json via tools.gems.eval_context (training-time render settings).

Every view-selection rule invocation is PRINTED and recorded in
RESULTS/figures/qual/manifest.json.

Usage:
    python -m tools.gems.report.qual_grids --gpu 5 [--scenes garden kitchen]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

EVAL_ROOT = "/data/peilincai/gems_stage1/eval"
OUT_DEFAULT = os.path.join(REPO_ROOT, "RESULTS", "figures", "qual")

# ---------------------------------------------------------------------------
# Frozen grid registry (LEDGER GOAL #014 Part B; banked rows only).
# Row order matters: clean FIRST (depth vmax reference), then B50, B25, B6R.
# b6r rows are optional: included iff their eval dir exists at run time.
# failure_view: (view_name_substring, provenance) or None -> argmin fallback.
# ---------------------------------------------------------------------------
GRIDS = {
    "garden": {
        "suite": "S-REND",
        "rows": [("clean", "garden_clean30k_v2"),
                 ("B50 B5", "garden_B50_importance_ft_e1v2"),
                 ("B25 B5", "garden_B25_importance_ft_e1v2")],
        "b6r": None,
        "failure_view": ("DSC08052", "E9 case12 (garden g3 fragmentation)"),
    },
    "kitchen": {
        "suite": "S-REND",
        "rows": [("clean", "kitchen_clean30k_v1"),
                 ("B50 B5", "kitchen_B50_importance_ft_s2"),
                 ("B25 B5", "kitchen_B25_importance_ft_s2")],
        "b6r": None,
        "failure_view": ("DSCF0768", "E9 case13 (indoor budget starvation)"),
    },
    "flowers": {
        "suite": "S-REND",
        "rows": [("clean", "flowers_clean30k_v1"),
                 ("B50 B5", "flowers_B50_importance_ft_s2"),
                 ("B25 B5", "flowers_B25_importance_ft_s2")],
        "b6r": None,
        "failure_view": ("_DSC9144", "E9 case01 (flowers compaction-floor fail)"),
    },
    "ss3dm_town01": {
        "suite": "S-GEO",
        "rows": [("clean", "ss3dm_town01_clean30k_geo_v1"),
                 ("B50 B5", "ss3dm_town01_B50_geo_v1"),
                 ("B25 B5", "ss3dm_town01_B25_importance_ft_s2")],
        "b6r": ("B6R", "b6r_ss3dm_town01_B50_v1"),
        "failure_view": None,  # no E9 case names a town01 view -> argmin rule
    },
    "ss3dm_town06": {
        "suite": "S-GEO",
        "rows": [("clean", "ss3dm_town06_clean30k_geo_v1"),
                 ("B50 B5", "ss3dm_town06_B50_geo_v1"),
                 ("B25 B5", "ss3dm_town06_B25_importance_ft_s2")],
        "b6r": None,
        "failure_view": ("front_00000088", "E9 case02 (town06 B50 S-GEO fail)"),
    },
    "toy_parking": {
        "suite": "S-GEO/S-DOWN (toy)",
        "rows": [("clean", "toy_parking_clean30k_v1"),
                 ("B50 B5", "toy_parking_B50_importance_ft_e1v2"),
                 ("B25 B5", "toy_parking_B25_importance_ft_e1v2")],
        "b6r": ("B6R", "e2r_toy_parking_B50_v1"),
        "failure_view": ("00035", "E9 case03/04/09 (train-coverage gap view)"),
    },
    "courtyard": {
        "suite": "dev_drive_A (courtyard)",
        "rows": [("clean", "courtyard_clean30k_v1"),
                 ("B50 B5", "courtyard_B50_importance_ft_e1v2"),
                 ("B25 B5", "courtyard_B25_importance_ft_e1v2")],
        "b6r": ("B6R", "e2r_courtyard_B50_v1"),
        "failure_view": ("DSC_0302", "E9 case08 (near-camera occluder view)"),
    },
}

CROP_STRIDE = 16  # px, sliding-window stride for the error-window search


def _load_row(eval_row: str) -> dict:
    with open(os.path.join(EVAL_ROOT, eval_row, "metrics.json")) as f:
        return json.load(f)


def _select_views(scene: str, cfg: dict, manifest_records: list):
    """Apply the frozen SS5 view-selection rules; print every invocation."""
    import numpy as np
    b50_row = cfg["rows"][1][1]
    m = _load_row(b50_row)
    arr_path = os.path.join(EVAL_ROOT, b50_row, "metrics.json")
    names = m["rendering"]["per_view"]["image_names"]
    psnr = np.asarray(m["rendering"]["per_view"]["psnr"], dtype=np.float64)
    n = len(names)

    def rec(rule, view, note):
        r = {"scene": scene, "rule": rule, "banked_array": arr_path,
             "banked_field": "rendering.per_view.psnr", "n_views": n,
             "chosen_view": view,
             "psnr_at_view_b50row": float(psnr[names.index(view)]),
             "note": note}
        manifest_records.append(r)
        print(f"[qual_grids][VIEW-SELECT] scene={scene} rule={rule} -> "
              f"view='{view}' (PSNR {r['psnr_at_view_b50row']:.3f} in banked "
              f"B50-B5 array {arr_path}, n={n}) {note}")
        return view

    best = rec("(a) best-case = argmax per-view PSNR",
               names[int(np.argmax(psnr))], "")
    med_idx = int(np.argsort(psnr)[(n - 1) // 2])
    median = rec("(b) median-PSNR = image_names[argsort(psnr)[(n-1)//2]]",
                 names[med_idx],
                 f"(sorted-rank {(n - 1) // 2} of {n})")
    if cfg["failure_view"] is not None:
        sub, prov = cfg["failure_view"]
        hits = [x for x in names if sub in x]
        assert len(hits) == 1, f"E9 failure view '{sub}' ambiguous/absent: {hits}"
        failure = rec("(c) failure = E9-taxonomy view", hits[0], f"[{prov}]")
    else:
        failure = rec("(c) failure = argmin per-view PSNR (no E9 case for "
                      "this scene)", names[int(np.argmin(psnr))], "")
    return {"best": best, "median": median, "failure": failure}


def _crop_window(err: "np.ndarray", mode: str):
    """Half-frame window minimizing (best) / maximizing (failure) mean error."""
    import numpy as np
    H, W = err.shape
    h, w = H // 2, W // 2
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(err, axis=0), axis=1)
    best_val, best_xy = None, (0, 0)
    for y0 in range(0, H - h + 1, CROP_STRIDE):
        for x0 in range(0, W - w + 1, CROP_STRIDE):
            s = ii[y0 + h, x0 + w] - ii[y0, x0 + w] - ii[y0 + h, x0] + ii[y0, x0]
            v = s / (h * w)
            if best_val is None or (v < best_val if mode == "min" else v > best_val):
                best_val, best_xy = v, (y0, x0)
    y0, x0 = best_xy
    return {"y0": int(y0), "x0": int(x0), "h": int(h), "w": int(w),
            "mean_err_in_window": float(best_val), "rule": f"arg{mode}-error "
            f"half-frame window, stride {CROP_STRIDE}, from B50-B5 row"}


def _render_row_views(ctx, views_by_name: dict, floater_ids):
    """Render the selected views for the CURRENT ctx.triangles model.

    Returns {view_name: {rgb: HxWx3 float32, depth: HxW float32 (<=0 invalid),
                         overlay: HxWx3 float32}}.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    out = {}
    for vname, cam in views_by_name.items():
        pkg = ctx.render_view(cam)
        rgb = pkg["render"].detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        depth = pkg["surf_depth"][0].detach().float().cpu().numpy()

        overlay = rgb.copy()
        if floater_ids is not None and len(floater_ids) > 0:
            ids_hr = pkg["rend_ids"][0].detach().long()
            fl = torch.as_tensor(np.asarray(floater_ids, dtype=np.int64),
                                 device=ids_hr.device)
            mask_hr = torch.isin(ids_hr, fl) & (ids_hr >= 0)
            depth_full = pkg.get("depth_full")
            if depth_full is not None:
                mask_hr &= depth_full[0].detach() > 0
            h, w = rgb.shape[:2]
            mask = F.interpolate(mask_hr[None, None].float(), size=(h, w),
                                 mode="area")[0, 0] > 0
            mask = mask.cpu().numpy()
            overlay[mask] = 0.4 * overlay[mask] + 0.6 * np.array([1.0, 0.0, 0.0])
        out[vname] = {"rgb": rgb, "depth": depth, "overlay": overlay}
    return out


def build_grid(scene: str, cfg: dict, out_dir: str, manifest: dict):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scene.triangle_model import TriangleModel
    from tools.gems.eval_context import build_eval_context
    from tools.gems.scenes import SCENES

    rows = list(cfg["rows"])
    if cfg["b6r"] is not None:
        lbl, row = cfg["b6r"]
        if os.path.isfile(os.path.join(EVAL_ROOT, row, "metrics.json")):
            rows.append((lbl, row))
            print(f"[qual_grids] {scene}: B6R row '{row}' present -> included")
        else:
            print(f"[qual_grids] {scene}: B6R row '{row}' absent -> skipped")

    sel_records = []
    views = _select_views(scene, cfg, sel_records)
    view_names = [views["best"], views["median"], views["failure"]]
    uniq_views = list(dict.fromkeys(view_names))

    # ---- render every model row at the selected views --------------------
    spec = SCENES[scene]
    panels, psnr_ann, row_meta = {}, {}, {}
    ctx = None
    gt_by_view = None
    for label, eval_row in rows:
        m = _load_row(eval_row)
        ckpt = m["checkpoint"]["path"]
        assert os.path.isfile(ckpt), f"banked checkpoint missing: {ckpt}"
        npz_path = os.path.join(EVAL_ROOT, eval_row, "geometry", "g3_floaters.npz")
        fl_ids = np.load(npz_path)["floater_tri_ids"]
        names = m["rendering"]["per_view"]["image_names"]
        pv = m["rendering"]["per_view"]["psnr"]
        psnr_ann[label] = {n: float(p) for n, p in zip(names, pv)}
        row_meta[label] = {"eval_row": eval_row, "checkpoint": ckpt,
                           "floater_npz": npz_path,
                           "n_floater_tri_ids": int(len(fl_ids)),
                           "psnr_mean_banked": m["rendering"]["mean"]["psnr"]}

        if ctx is None:
            print(f"[qual_grids] {scene}: building eval context ({label})")
            ctx = build_eval_context(ckpt, spec)
            cams = {c.image_name: c for c in ctx.test_cams}
            for v in uniq_views:
                assert v in cams, f"view '{v}' not in test cams of {scene}"
            views_by_name = {v: cams[v] for v in uniq_views}
            gt_by_view = {
                v: cams[v].original_image[:3].detach().float().clamp(0, 1)
                .cpu().numpy().transpose(1, 2, 0) for v in uniq_views}
        else:
            print(f"[qual_grids] {scene}: swapping model -> {label}")
            del ctx.triangles
            torch.cuda.empty_cache()
            tri = TriangleModel(3)
            tri.scaling = 4  # training-time supersampling (eval_context)
            tri.load_parameters(os.path.dirname(ckpt), device="cuda")
            ctx.triangles = tri

        panels[label] = _render_row_views(ctx, views_by_name, fl_ids)

    del ctx
    torch.cuda.empty_cache()

    # ---- frozen crop rule: windows from the B50 B5 row -------------------
    b50_label = rows[1][0]
    crops = {}
    for block, vname, mode in (("best", views["best"], "min"),
                               ("failure", views["failure"], "max")):
        err = np.abs(panels[b50_label][vname]["rgb"] - gt_by_view[vname]).mean(axis=-1)
        crops[block] = _crop_window(err, mode)
    crops["median"] = None  # full frame by frozen rule

    # depth vmax per view from the CLEAN row (rows[0])
    clean_label = rows[0][0]
    vmax_by_view = {}
    for v in uniq_views:
        d = panels[clean_label][v]["depth"]
        valid = d > 0
        vmax_by_view[v] = float(np.quantile(d[valid], 0.99)) if valid.any() else 1.0

    # ---- compose the figure ----------------------------------------------
    def _apply(img, win):
        if win is None:
            return img
        return img[win["y0"]:win["y0"] + win["h"], win["x0"]:win["x0"] + win["w"]]

    blocks = [("(a) BEST", "best"), ("(b) MEDIAN", "median"),
              ("(c) FAILURE", "failure")]
    nr, nc = len(rows), 9
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.15 * nr),
                             squeeze=False)
    for bi, (btitle, bkey) in enumerate(blocks):
        vname = views[bkey]
        win = crops[bkey]
        for ri, (label, _) in enumerate(rows):
            p = panels[label][vname]
            axr = axes[ri][3 * bi]
            axr.imshow(_apply(p["rgb"], win))
            axr.text(0.02, 0.04, f"{psnr_ann[label].get(vname, float('nan')):.2f} dB",
                     transform=axr.transAxes, fontsize=7, color="w",
                     bbox=dict(facecolor="k", alpha=0.55, pad=1.5))
            d = _apply(p["depth"], win)
            axes[ri][3 * bi + 1].imshow(
                np.where(d > 0, d, np.nan), cmap="turbo",
                vmin=0.0, vmax=vmax_by_view[vname])
            axes[ri][3 * bi + 2].imshow(_apply(p["overlay"], win))
            if bi == 0:
                axes[ri][0].set_ylabel(label, fontsize=11)
        crop_note = "full frame" if win is None else \
            f"crop [{win['y0']}:{win['y0']+win['h']},{win['x0']}:{win['x0']+win['w']}]"
        axes[0][3 * bi].set_title(f"{btitle} '{vname}'\n{crop_note} | RGB",
                                  fontsize=8)
        axes[0][3 * bi + 1].set_title("median depth", fontsize=8)
        axes[0][3 * bi + 2].set_title("floater overlay", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ri, (label, _) in enumerate(rows):
        axes[ri][0].set_ylabel(label, fontsize=11)
    fig.suptitle(
        f"{scene} ({cfg['suite']}) — E11-QUAL grid. View selection per Stage-2 "
        f"prompt SS5 (script-chosen; see manifest.json). Crop windows from the "
        f"{b50_label} row error map, applied identically to all rows; depth "
        f"vmax = P99 of clean-row depth per view.", fontsize=9)
    fig.tight_layout(rect=(0.01, 0, 1, 0.96))
    png = os.path.join(out_dir, f"{scene}_qual_grid.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"[qual_grids] wrote {png}")

    manifest["scenes"][scene] = {
        "suite": cfg["suite"],
        "view_selection": sel_records,
        "views": views,
        "crop_windows": crops,
        "depth_vmax_by_view": vmax_by_view,
        "rows": row_meta,
        "output_png": png,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--scenes", nargs="*", default=list(GRIDS.keys()))
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    prior_scenes = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            prior_scenes = json.load(f).get("scenes", {})
    manifest = {"generator": "tools/gems/report/qual_grids.py",
                "frozen_rules": {
                    "best": "argmax banked per-view PSNR of the scene's B50 B5 row",
                    "median": "image_names[argsort(psnr)[(n-1)//2]] of the same "
                              "banked array (script-chosen)",
                    "failure": "E9-taxonomy view where a case names one, else "
                               "argmin banked per-view PSNR",
                    "crops": "best->argmin-error half-frame window / failure->"
                             "argmax-error half-frame window (stride "
                             f"{CROP_STRIDE}px) computed on the B50 B5 row only, "
                             "applied identically to all rows; median->full frame",
                    "depth_vmax": "P99 of CLEAN-row valid depth per view",
                    "floater_overlay": "each row tinted by its OWN banked "
                                       "g3_floaters.npz floater_tri_ids",
                },
                "eval_root": EVAL_ROOT,
                "scenes": prior_scenes}
    for scene in args.scenes:
        build_grid(scene, GRIDS[scene], args.out, manifest)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"[qual_grids] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
