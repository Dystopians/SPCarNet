"""GEMS B3 baseline — importance-free geometric decimation (QEM-style) + safe FT.

MATRIX cell B3 (LEDGER GOAL #013, pre-registered): quadric-edge-collapse
decimation of a clean checkpoint to keep_count = floor(budget * T_clean)
faces, per-vertex attribute transfer by nearest-ORIGINAL-vertex, then the
validated safe fine-tune (features-only) + run_eval — mirroring
tools/gems/gems_pipeline.py stage-for-stage so B3 rows are protocol-identical
to the B2/B4/B5 rows except for the prune rule.

    source checkpoint -> QEM decimate -> attribute transfer -> rebuilt
    point_cloud_state_dict.pt (checkpoint_compaction output contract)
    -> [safe FT 10k, features-only] -> run_eval.py -> row.json

Decimation backend (task-spec fallback order: open3d -> trimesh -> pymeshlab;
installing anything is FORBIDDEN — if no backend imports, this tool exits with
an INFEASIBLE report instead of running):
  open3d 0.19.0 IS importable but was VERIFIED NON-FUNCTIONAL on these
  checkpoints (pre-run smoke, LEDGER GOAL #013 amendment): the models are
  heavily non-manifold triangle soup (toy edge census: 44% of edges have >=3
  incident faces, up to 8) and open3d's manifold-edge-collapse QEM stalls at
  5.5% removal (6,590,546 -> 6,231,952 faces vs target 3,295,279). Fallback #2
  is therefore used: trimesh's QEM engine — trimesh.Trimesh.
  simplify_quadric_decimation is a thin wrapper around fast_simplification
  (installed, 0.1.13; sp4cerat Fast-Quadric-Mesh-Simplification, soup-
  tolerant). We call fast_simplification.simplify directly (same engine,
  no Trimesh object munging), at the library-default aggression 7, in
  iterative rounds (each round re-seeds the collapse queue; a single round
  hits its internal iteration cap ~21% above target on toy) until
  keep_count is reached. Deterministic (no RNG).

CRITICAL contract note: decimation creates NEW vertex positions, so per-vertex
features (features_dc, features_rest) and vertex_weight cannot be index-mapped
as in checkpoint_compaction — they are transferred from the nearest original
(finite) vertex via scipy cKDTree. Face-level stat tensors
(importance_score / image_size / pixel_count) are rebuilt as zeros of the new
face count (load_parameters re-zeros them at load anyway); sigma /
active_sh_degree / opacity_floor are copied verbatim.

Usage:
    $PY -m tools.gems_train.qem_prune \
        --scene <garden|toy_parking|courtyard|...> \
        --source-ckpt <point_cloud_state_dict.pt> \
        --budget 0.5 --tag b3 [--ft-iters 10000] [--gpu N] \
        [--skip-eval] [--stop-after qem|finetune]

D4 GUARDS: the QEM stage touches geometry only (no images at all); the render
verification uses tools/gems/triangle_evidence._TrainOnlyContext (TRAIN view
0); the FT stage reuses gems_pipeline.build_train_cmd/stage_finetune with the
frozen safe-FT flags; this tool never reads test GT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PIPELINE_VERSION = "gems_qem_prune_v1"
MODE = "qem_ft"  # B3: decimate + safe FT (features-only)
# Frozen safe-FT channel (E1 variant 2, LEDGER GOAL #005): features at the
# default lr, positions and weights frozen. Identical to the banked
# B5 (importance_ft e1v2 / s2) and B2 (random_ft) rows.
SAFE_FT_LR_OVERRIDES = {"feature_lr": None,          # default 0.0016
                        "weight_lr": 0.0,
                        "lr_triangles_points_init": 0.0}
BUDGET_TOLERANCE = 0.02  # fair-budget matching rule (MATRIX section 4): +/-2%


QEM_AGGRESSION = 7   # fast_simplification library default ("slow and good" end)
QEM_MAX_ROUNDS = 8   # each round re-seeds the collapse queue on the smaller mesh


def _select_backend():
    """Return (name, module) for the working decimation backend.
    Task-spec order open3d -> trimesh -> pymeshlab; open3d is functionally
    ruled out on these non-manifold soups (see module docstring), so the
    working backend is trimesh's engine, fast_simplification."""
    try:
        import fast_simplification  # trimesh.simplify_quadric_decimation's engine
        return "fast_simplification", fast_simplification
    except Exception:
        pass
    try:
        import pymeshlab  # noqa: F401
        return "pymeshlab", pymeshlab
    except Exception:
        pass
    return None, None


def stage_qem(scene: str, spec, source_ckpt: str, budget: float,
              keep_count: int, t_clean: int, model_dir: str,
              stamps, cfg_hash: str) -> dict:
    import numpy as np
    import torch
    from ss3dm_prior.meshsplatopt.checkpoint_compaction import (
        VERTEX_KEYS, FACE_KEYS, copy_model_metadata, validate_faces)
    from tools.gems.gems_pipeline import _source_model_dir_and_iter

    backend, _ = _select_backend()
    if backend is None:
        raise SystemExit(
            "[qem] INFEASIBLE: no decimation backend importable "
            "(open3d ruled out on non-manifold soup; trimesh/"
            "fast_simplification and pymeshlab absent) and installing is "
            "forbidden. File the INFEASIBLE note.")
    assert backend == "fast_simplification", (
        f"backend '{backend}' reached but this run pre-registered "
        "fast_simplification (trimesh's QEM engine, verified functional); "
        "refusing a silent backend swap")
    import fast_simplification as fs

    src_model, load_iter = _source_model_dir_and_iter(source_ckpt)
    state = torch.load(source_ckpt, map_location="cpu")
    verts = state["triangles_points"].detach().to(torch.float32).cpu().numpy()
    faces = state["_triangle_indices"].detach().cpu().numpy().astype(np.int64)
    assert faces.shape[0] == t_clean

    # --- exclude non-finite geometry from the decimation input (e.g. the 13
    # NaN faces in toy_parking clean30k; PROTOCOL 4.3 non-finite exclusion).
    finite_vertex = np.isfinite(verts).all(axis=1)
    face_finite = finite_vertex[faces].all(axis=1)
    n_nonfinite_faces = int((~face_finite).sum())
    faces_in = faces[face_finite]
    assert keep_count <= faces_in.shape[0], (
        f"keep_count {keep_count} exceeds finite face count {faces_in.shape[0]}")

    backend_desc = (f"trimesh-engine fast_simplification "
                    f"{getattr(fs, '__version__', '?')} quadric edge collapse, "
                    f"agg={QEM_AGGRESSION}, iterative")
    print(f"[qem] backend={backend_desc} | T_clean={t_clean} "
          f"(finite {faces_in.shape[0]}, nonfinite excluded {n_nonfinite_faces}) "
          f"-> target {keep_count}", flush=True)

    # --- QEM decimation (quadric edge collapse), iterative rounds.
    t0 = time.time()
    new_v = verts.astype(np.float64)
    new_f = faces_in
    rounds = []
    for rnd in range(QEM_MAX_ROUNDS):
        if new_f.shape[0] <= keep_count:
            break
        pre = int(new_f.shape[0])
        new_v, new_f = fs.simplify(new_v, new_f.astype(np.int64),
                                   target_count=int(keep_count),
                                   agg=QEM_AGGRESSION)
        rounds.append({"round": rnd + 1, "pre_faces": pre,
                       "post_faces": int(new_f.shape[0]),
                       "post_vertices": int(new_v.shape[0])})
        print(f"[qem] round {rnd + 1}: {pre} -> {new_f.shape[0]} faces "
              f"({new_v.shape[0]} verts)", flush=True)
        if int(new_f.shape[0]) == pre:
            raise RuntimeError(
                f"QEM stalled at {pre} faces (target {keep_count}) — "
                "no progress in a full round; refusing to ship an "
                "off-budget B3 row")
    dt_dec = time.time() - t0
    t_new, v_new = int(new_f.shape[0]), int(new_v.shape[0])
    rel_err = (t_new - keep_count) / float(keep_count)
    print(f"[qem] decimated to {t_new} faces / {v_new} vertices in "
          f"{dt_dec / 60.0:.1f} min over {len(rounds)} rounds "
          f"(target {keep_count}, rel err {rel_err:+.4%})", flush=True)
    # Fair-budget matching (MATRIX section 4): within +/-2% of keep_count, and
    # never ABOVE the hard budget (PROTOCOL section 2 floor semantics).
    assert t_new <= keep_count, (
        f"decimated count {t_new} exceeds hard budget {keep_count}")
    assert abs(rel_err) <= BUDGET_TOLERANCE, (
        f"decimated count {t_new} outside +/-2% of {keep_count}")
    assert np.isfinite(new_v).all(), "decimated vertices contain non-finite values"

    # --- per-vertex attribute transfer: nearest ORIGINAL finite vertex.
    from scipy.spatial import cKDTree
    t0 = time.time()
    orig_ids = np.nonzero(finite_vertex)[0]
    tree = cKDTree(verts[finite_vertex].astype(np.float64))
    nn_dist, nn = tree.query(new_v, k=1, workers=-1)
    src_ids = torch.from_numpy(orig_ids[nn])
    print(f"[qem] KDTree transfer: {v_new} queries in {time.time() - t0:.0f}s "
          f"(nn dist mean {float(nn_dist.mean()):.4g}, "
          f"p99 {float(np.percentile(nn_dist, 99)):.4g})", flush=True)

    # --- rebuild the checkpoint (checkpoint_compaction output contract).
    faces_dtype = state["_triangle_indices"].dtype
    out: dict = {}
    for key, value in state.items():
        if torch.is_tensor(value) and key == "_triangle_indices":
            out[key] = torch.from_numpy(new_f).to(dtype=faces_dtype).clone()
        elif torch.is_tensor(value) and key == "triangles_points":
            out[key] = torch.from_numpy(new_v).to(
                dtype=state["triangles_points"].dtype).clone()
        elif torch.is_tensor(value) and key in VERTEX_KEYS and value.shape[0] == verts.shape[0]:
            out[key] = value.detach().cpu()[src_ids].clone()
        elif torch.is_tensor(value) and key in FACE_KEYS and value.shape[0] == t_clean:
            out[key] = torch.zeros((t_new,) + tuple(value.shape[1:]),
                                   dtype=value.dtype)
        elif torch.is_tensor(value):
            raise AssertionError(
                f"unhandled tensor key '{key}' shape {tuple(value.shape)} — "
                "checkpoint contract changed; refusing to guess")
        else:
            out[key] = value
    del state

    out_ckpt_dir = os.path.join(model_dir, "point_cloud", f"iteration_{load_iter}")
    os.makedirs(out_ckpt_dir, exist_ok=True)
    pruned_ckpt = os.path.join(out_ckpt_dir, "point_cloud_state_dict.pt")
    copy_model_metadata(src_model, model_dir)
    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
    assert invalid == 0, f"rebuilt ckpt has {invalid} invalid face indices"
    torch.save(out, pruned_ckpt)

    audit = {
        "selector_mode": f"gems_qem_{MODE}_B{int(round(budget * 100))}",
        "backend": backend_desc,
        "rounds": rounds,
        "source_checkpoint": source_ckpt,
        "output_checkpoint": pruned_ckpt,
        "iteration": load_iter,
        "pre_triangles": t_clean,
        "pre_triangles_finite": int(faces_in.shape[0]),
        "nonfinite_faces_excluded": n_nonfinite_faces,
        "target_keep_count": keep_count,
        "post_triangles": t_new,
        "budget_rel_err": rel_err,
        "pre_vertices": int(verts.shape[0]),
        "post_vertices": v_new,
        "degenerate_face_count": degenerate,
        "invalid_index_count": invalid,
        "attr_transfer": "nearest_original_finite_vertex_cKDTree",
        "nn_dist_mean": float(nn_dist.mean()),
        "nn_dist_p99": float(np.percentile(nn_dist, 99)),
        "decimation_seconds": dt_dec,
    }
    with open(os.path.join(model_dir, "topology_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # --- VERIFY: loads in TriangleModel and renders one TRAIN view.
    from tools.gems.triangle_evidence import _TrainOnlyContext
    vctx = _TrainOnlyContext(pruned_ckpt, spec)
    n_loaded = int(vctx.faces().shape[0])
    assert n_loaded == t_new, f"loaded {n_loaded} != decimated {t_new}"
    pkg = vctx.render_view(vctx.train_cams[0])
    img = pkg["render"].detach()
    assert img.dim() == 3 and img.shape[0] == 3 and torch.isfinite(img).all(), (
        "decimated checkpoint render sanity check failed")
    audit["verified_render_train_view"] = vctx.train_cams[0].image_name
    del pkg, img, vctx
    torch.cuda.empty_cache()

    payload = {"pruned_ckpt": pruned_ckpt, "load_iteration": load_iter,
               "t_clean": t_clean, "keep_count": keep_count, "audit": audit,
               "keep_rule": "qem_fast_simplification_quadric_edge_collapse_v1"}
    stamps.write("qem", cfg_hash, payload)
    print(f"[qem] QEM stage OK: {t_clean} -> {t_new} triangles "
          f"({audit['backend']}); render verified on train view "
          f"{audit['verified_render_train_view']}", flush=True)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="GEMS B3 QEM decimation baseline")
    p.add_argument("--scene", required=True)
    p.add_argument("--source-ckpt", required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ft-iters", type=int, default=10000)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--stop-after", choices=["qem", "finetune"], default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # before torch import

    from tools.gems.gems_pipeline import (
        MODELS_ROOT, EVAL_ROOT, StageStamps, budget_label, build_train_cmd,
        git_commit, sha256_str, stage_eval, stage_finetune, stage_preflight,
        utc_now, _ckpt_shapes, _source_model_dir_and_iter)
    from tools.gems.scenes import SCENES
    from tools.gems.triangle_evidence import checkpoint_fingerprint

    assert 0.0 < args.budget <= 1.0
    spec = SCENES[args.scene]
    blabel = budget_label(args.budget)
    name = f"{args.scene}_{blabel}_{MODE}_{args.tag}"
    run_name = f"gems_{name}"
    model_dir = os.path.join(MODELS_ROOT, name)
    eval_dir = os.path.join(EVAL_ROOT, name)
    os.makedirs(model_dir, exist_ok=True)
    stamps = StageStamps(os.path.join(model_dir, "gems_stages"))

    source_ckpt = os.path.abspath(args.source_ckpt)
    _, load_iter = _source_model_dir_and_iter(source_ckpt)
    ckpt_fp = checkpoint_fingerprint(source_ckpt)
    t_clean = _ckpt_shapes(source_ckpt)["triangles"]
    keep_count = int(args.budget * t_clean)  # floor — same rule as gems_pipeline
    final_iter = load_iter + int(args.ft_iters)
    final_ckpt = os.path.join(model_dir, "point_cloud", f"iteration_{final_iter}",
                              "point_cloud_state_dict.pt")

    train_cmd = build_train_cmd(spec, model_dir, load_iter, final_iter,
                                run_name, SAFE_FT_LR_OVERRIDES)
    import shlex
    stage_cfg = {
        "preflight": f"{PIPELINE_VERSION}|preflight|roots={MODELS_ROOT},{EVAL_ROOT}|min_free_gb=50",
        "qem": (f"{PIPELINE_VERSION}|qem|scene={args.scene}"
                f"|ckpt={ckpt_fp['sha256_first16mb']}|budget={args.budget}"
                f"|t_clean={t_clean}|keep_count={keep_count}"
                f"|backend=fast_simplification_qem_agg{QEM_AGGRESSION}_iterative"
                f"|transfer=nearest_original_finite_vertex"),
        "finetune": f"{PIPELINE_VERSION}|finetune|{shlex.join(train_cmd)}",
        "eval": (f"{PIPELINE_VERSION}|eval|scene={args.scene}"
                 f"|checkpoint={final_ckpt}|out={eval_dir}"),
    }
    stage_hash = {k: sha256_str(v) for k, v in stage_cfg.items()}
    config_hash = sha256_str("\n".join(
        f"{k}::{stage_cfg[k]}" for k in ("preflight", "qem", "finetune", "eval")))

    print(f"[qem] run={name} T_clean={t_clean} keep_count={keep_count} "
          f"config_hash={config_hash[:12]}", flush=True)

    def run_stage(stage_name, fn, *fn_args):
        h = stage_hash[stage_name]
        if stamps.is_done(stage_name, h):
            print(f"[qem] stage '{stage_name}' already complete — skipping",
                  flush=True)
            return stamps.load(stage_name)["payload"]
        return fn(*fn_args)

    stage_preflight(stamps, stage_hash["preflight"])

    qem_payload = run_stage("qem", stage_qem, args.scene, spec, source_ckpt,
                            args.budget, keep_count, t_clean, model_dir,
                            stamps, stage_hash["qem"])
    pruned_ckpt = qem_payload["pruned_ckpt"]
    if args.stop_after == "qem":
        print("[qem] --stop-after qem: done", flush=True)
        return

    ft_payload = run_stage("finetune", stage_finetune, train_cmd, model_dir,
                           pruned_ckpt, final_iter, args.gpu, stamps,
                           stage_hash["finetune"])
    if args.stop_after == "finetune":
        print("[qem] --stop-after finetune: done", flush=True)
        return

    eval_payload = None
    if not args.skip_eval:
        eval_payload = run_stage("eval", stage_eval, args.scene, final_ckpt,
                                 eval_dir, args.gpu, stamps, stage_hash["eval"])

    row = {
        "scene": args.scene,
        "budget": args.budget,
        "budget_label": blabel,
        "mode": MODE,
        "tag": args.tag,
        "config_hash": config_hash,
        "git_commit": git_commit(),
        "source_ckpt": ckpt_fp,
        "n_triangles_clean": t_clean,
        "n_triangles_pruned": qem_payload["audit"]["post_triangles"],
        "target_keep_count": keep_count,
        "prune_rule": qem_payload["keep_rule"],
        "final_ckpt": final_ckpt,
        "ft_iters": int(args.ft_iters),
        "ft_wallclock_min": (ft_payload or {}).get("ft_wallclock_min"),
        "metrics_json": (eval_payload or {}).get("metrics_json"),
        "eval_skipped": bool(args.skip_eval),
        "wandb_name": run_name,
        "written_utc": utc_now(),
    }
    os.makedirs(eval_dir, exist_ok=True)
    row_path = os.path.join(eval_dir, "row.json")
    with open(row_path, "w") as f:
        json.dump(row, f, indent=1)
    stamps.write("row", sha256_str(config_hash), {"row_json": row_path})
    print(f"[qem] DONE {name}\n[qem] row: {row_path}", flush=True)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
