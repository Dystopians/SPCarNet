"""GEMS M3 — E2 mechanism variant 3/3 (HUMAN-SANCTIONED): evidence-based
floater REMOVAL (LEDGER.md GOAL #008, Pre-registration C).

Mechanism (exactly as pre-registered):
  1. On the M2 B50 model, run the PROTOCOL.md §4.3 g3 support pass — per-
     triangle train-support from `rend_ids` over <= 60 evenly spaced TRAIN
     views — plus vertex-graph connected components. A component is a FLOATER
     iff every member triangle has support <= 1 AND the component has
     < 10,000 triangles. The g3 machinery is CRIBBED (imported) from
     tools/gems/geometry_metrics.py, the verified implementation; nothing is
     re-derived here.
  2. Removal mask = ALL triangles of floater components.
  3. Removal is applied through the same keep-mask compaction call
     gems_pipeline.stage_prune uses
     (ss3dm_prior/meshsplatopt/checkpoint_compaction.apply_compaction:
     tensor surgery + vertex GC + index remap). Triangle count only
     DECREASES, so the model stays <= budget B. The pruned checkpoint is
     verified to load in TriangleModel and render one TRAIN view.
  4. Features-only fine-tune 5k iters — the E1-v2-validated safe channel —
     via train.py with the EXACT flag pattern of gems_pipeline's finetune
     stage (build_train_cmd + stage_finetune guards): topology frozen,
     --test_iterations -1, --seed 0, --weight_lr 0.0
     --lr_triangles_points_init 0.0, WANDB_MODE=online.
  5. run_eval.py (THE single mouth, D5) on the final checkpoint.

D4 purity: the removal decision consumes ONLY train-view renders (rend_ids
support) and mesh topology — no GT of any kind, no test cameras
(_TrainOnlyContext never image-loads test views).

Usage:
    $PY -m tools.gems.floater_prune \
        --scene <toy_parking|courtyard> \
        --checkpoint <M2 B50 point_cloud_state_dict.pt> \
        --out-model-dir <dir> [--ft-iters 5000] [--gpu N] [--eval-out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable

GEMS_ROOT = "/data/peilincai/gems_stage1"
EVAL_ROOT = os.path.join(GEMS_ROOT, "eval")

TOOL_VERSION = "floater_prune_v1"


def parse_args():
    p = argparse.ArgumentParser(description="GEMS E2 variant 3: g3 floater removal")
    p.add_argument("--scene", required=True, choices=["toy_parking", "courtyard"])
    p.add_argument("--checkpoint", required=True,
                   help="M2 B50 point_cloud_state_dict.pt (source model)")
    p.add_argument("--out-model-dir", required=True,
                   help="output model dir (pruned ckpt + fine-tune land here)")
    p.add_argument("--ft-iters", type=int, default=5000)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--eval-out", default=None,
                   help="run_eval output dir (default: "
                        f"{EVAL_ROOT}/e2v3_<scene>_B50_v1)")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def stage_floater_prune(scene: str, spec, source_ckpt: str, src_model: str,
                        load_iter: int, out_model: str, stamps, cfg_hash: str) -> dict:
    """Steps 1-3: g3 floater pass, removal mask, compaction, load+render verify."""
    import numpy as np
    import torch
    # CRIB the verified g3 machinery — the PROTOCOL definition lives there.
    from tools.gems.geometry_metrics import (
        G3_FLOATER_MAX_SIZE,
        G3_FLOATER_MAX_SUPPORT,
        G3_MAX_SUPPORT_VIEWS,
        _finite_faces_mask,
        _mesh_component_labels,
        _support_counts,
    )
    from tools.gems.triangle_evidence import _TrainOnlyContext
    from ss3dm_prior.meshsplatopt.checkpoint_compaction import apply_compaction

    # ---- (1) g3 support pass + components on the SOURCE model (TRAIN only)
    t0 = time.time()
    ctx = _TrainOnlyContext(source_ckpt, spec)
    faces_np = ctx.faces().detach().cpu().numpy().astype(np.int64)
    n_vertices = int(ctx.vertices().shape[0])
    n_triangles = int(faces_np.shape[0])
    assert n_triangles > 0, "source checkpoint has no triangles"
    assert n_triangles < 2 ** 24, (
        f"n_triangles={n_triangles} >= 2^24 exceeds rend_ids float32 precision")

    finite_mask = _finite_faces_mask(ctx, faces_np)
    face_labels = _mesh_component_labels(faces_np, n_vertices)
    support, view_indices = _support_counts(ctx, n_triangles,
                                            max_views=G3_MAX_SUPPORT_VIEWS)

    # ---- EXACT compute_g3 reduction (PROTOCOL §4.3): floater component =
    # every member triangle support <= 1 AND component size < 10,000.
    n_labels = int(face_labels.max()) + 1
    fin_labels = face_labels[finite_mask]
    comp_sizes = np.bincount(fin_labels, minlength=n_labels)
    comp_max_support = np.zeros(n_labels, dtype=np.int64)
    np.maximum.at(comp_max_support, fin_labels, support[finite_mask].astype(np.int64))
    has_faces = comp_sizes > 0
    is_floater = (
        has_faces
        & (comp_max_support <= G3_FLOATER_MAX_SUPPORT)
        & (comp_sizes < G3_FLOATER_MAX_SIZE)
    )
    # ---- (2) removal mask = all floater-component triangles
    floater_face_mask = is_floater[face_labels] & finite_mask
    drop = np.nonzero(floater_face_mask)[0].astype(np.int64)
    n_floater_comps = int(is_floater.sum())
    frac = float(floater_face_mask.mean())
    print(f"[floater_prune] {scene}: {n_triangles} triangles, "
          f"{int(has_faces.sum())} components, "
          f"{n_floater_comps} FLOATER components -> removing {drop.shape[0]} "
          f"triangles ({frac:.6f} fraction) "
          f"[support views: {int(view_indices.shape[0])}, "
          f"g3 pass {time.time() - t0:.1f}s]", flush=True)

    # Durable record of the decision inputs (train-only evidence).
    npz_path = os.path.join(out_model, "floater_prune_g3.npz")
    comp_ids = np.nonzero(has_faces)[0]
    np.savez_compressed(
        npz_path,
        floater_tri_ids=drop,
        component_sizes=comp_sizes[comp_ids].astype(np.int64),
        component_max_support=comp_max_support[comp_ids],
        component_is_floater=is_floater[comp_ids],
        triangle_support=support.astype(np.int32),
        support_view_indices=view_indices.astype(np.int64),
    )

    del ctx
    torch.cuda.empty_cache()

    # ---- (3) removal via the SAME compaction call gems_pipeline.stage_prune
    # uses (apply_compaction takes the REMOVE list; keep-mask surgery +
    # vertex GC + index remap inside).
    audit = apply_compaction(
        src_model, out_model, load_iter, drop,
        selector_mode=f"gems_e2v3_floater_prune_{scene}_B50",
    )
    assert audit.post_triangles == n_triangles - int(drop.shape[0]), (
        f"post_triangles {audit.post_triangles} != "
        f"{n_triangles} - {drop.shape[0]}")
    assert audit.invalid_index_count == 0, "pruned ckpt has invalid face indices"

    pruned_ckpt = os.path.join(
        out_model, "point_cloud", f"iteration_{load_iter}",
        "point_cloud_state_dict.pt")
    assert os.path.isfile(pruned_ckpt), pruned_ckpt

    # ---- VERIFY: loads in TriangleModel and renders one TRAIN view
    # (mirrors gems_pipeline.stage_prune's verification block).
    vctx = _TrainOnlyContext(pruned_ckpt, spec)
    n_loaded = int(vctx.faces().shape[0])
    assert n_loaded == audit.post_triangles, (
        f"loaded pruned ckpt has {n_loaded} triangles != {audit.post_triangles}")
    pkg = vctx.render_view(vctx.train_cams[0])
    img = pkg["render"].detach()
    assert img.dim() == 3 and img.shape[0] == 3 and torch.isfinite(img).all(), (
        "pruned checkpoint render sanity check failed")
    verify_view = vctx.train_cams[0].image_name
    del pkg, img, vctx
    torch.cuda.empty_cache()
    print(f"[floater_prune] pruned ckpt verified: loads + renders train view "
          f"'{verify_view}' ({n_loaded} triangles)", flush=True)

    payload = {
        "pruned_ckpt": pruned_ckpt,
        "load_iteration": load_iter,
        "n_triangles_source": n_triangles,
        "n_components": int(has_faces.sum()),
        "floater_component_count": n_floater_comps,
        "removed_triangles": int(drop.shape[0]),
        "removed_fraction": frac,
        "n_support_views": int(view_indices.shape[0]),
        "g3_constants": {
            "max_support": int(G3_FLOATER_MAX_SUPPORT),
            "max_component_size": int(G3_FLOATER_MAX_SIZE),
            "max_views": int(G3_MAX_SUPPORT_VIEWS),
        },
        "g3_npz": npz_path,
        "audit": audit.to_dict(),
        "verified_render_train_view": verify_view,
    }
    stamps.write("floater_prune", cfg_hash, payload)
    return payload


def main():
    args = parse_args()
    if args.gpu is not None:
        # must happen before any torch import in this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from tools.gems.scenes import SCENES
    from tools.gems.triangle_evidence import checkpoint_fingerprint
    from tools.gems.gems_pipeline import (
        StageStamps,
        _source_model_dir_and_iter,
        build_train_cmd,
        sha256_str,
        stage_eval,
        stage_finetune,
    )
    from tools.storage_preflight import volume_report

    spec = SCENES[args.scene]
    source_ckpt = os.path.abspath(args.checkpoint)
    src_model, load_iter = _source_model_dir_and_iter(source_ckpt)
    out_model = os.path.abspath(args.out_model_dir)
    os.makedirs(out_model, exist_ok=True)
    eval_dir = args.eval_out or os.path.join(EVAL_ROOT, f"e2v3_{args.scene}_B50_v1")
    run_name = f"gems_e2v3_{args.scene}_B50"
    final_iter = load_iter + int(args.ft_iters)

    # ---- storage preflight (D6): out-model volume + eval volume, 50 GB floor
    reports = [volume_report(p, 50.0) for p in (out_model, EVAL_ROOT)]
    if not all(r["ok"] for r in reports):
        raise SystemExit(f"[floater_prune] PREFLIGHT FAIL (D6): {json.dumps(reports)}")
    print(f"[floater_prune] preflight OK: "
          f"{[(r['path'], r['free_gb']) for r in reports]}", flush=True)

    ckpt_fp = checkpoint_fingerprint(source_ckpt)
    stamps = StageStamps(os.path.join(out_model, "gems_stages"))

    # ---- fine-tune command: EXACT gems_pipeline flag pattern; features-only
    # (weight_lr=0, position lr=0); only the wandb group/name differ (E2 tag).
    train_cmd = build_train_cmd(
        spec, out_model, load_iter, final_iter, run_name,
        lr_overrides={"weight_lr": 0.0, "lr_triangles_points_init": 0.0},
    )
    gi = train_cmd.index("--wandb_group")
    train_cmd[gi + 1] = "gems_m3_e2"

    final_ckpt = os.path.join(
        out_model, "point_cloud", f"iteration_{final_iter}",
        "point_cloud_state_dict.pt")

    # ---- per-stage config hashes (D6 resumability, gems_pipeline convention)
    stage_cfg = {
        "floater_prune": (
            f"{TOOL_VERSION}|prune|scene={args.scene}"
            f"|ckpt={ckpt_fp['sha256_first16mb']}"
            f"|g3=support<=1,size<10000,views<=60|load_iter={load_iter}"),
        "finetune": f"{TOOL_VERSION}|finetune|{shlex.join(train_cmd)}",
        "eval": (f"{TOOL_VERSION}|eval|scene={args.scene}"
                 f"|checkpoint={final_ckpt}|out={eval_dir}"),
    }
    stage_hash = {k: sha256_str(v) for k, v in stage_cfg.items()}

    def run_stage(stage_name, fn, *fn_args):
        h = stage_hash[stage_name]
        if stamps.is_done(stage_name, h):
            print(f"[floater_prune] stage '{stage_name}' already complete "
                  f"(hash match) — skipping", flush=True)
            return stamps.load(stage_name)["payload"]
        return fn(*fn_args)

    # ---- (1)-(3) floater pass + removal + verify
    prune_payload = run_stage(
        "floater_prune", stage_floater_prune, args.scene, spec, source_ckpt,
        src_model, load_iter, out_model, stamps, stage_hash["floater_prune"])
    pruned_ckpt = prune_payload["pruned_ckpt"]

    # ---- (4) features-only FT (gems_pipeline stage_finetune: flag guards,
    # WANDB_MODE=online, post-FT topology audit)
    ft_payload = run_stage(
        "finetune", stage_finetune, train_cmd, out_model, pruned_ckpt,
        final_iter, args.gpu, stamps, stage_hash["finetune"])

    assert os.path.isfile(final_ckpt), f"final checkpoint missing: {final_ckpt}"

    # ---- (5) run_eval.py (THE single mouth)
    eval_payload = None
    if args.skip_eval:
        print("[floater_prune] --skip-eval: eval stage skipped", flush=True)
    else:
        eval_payload = run_stage("eval", stage_eval, args.scene, final_ckpt,
                                 eval_dir, args.gpu, stamps, stage_hash["eval"])

    row = {
        "tool": TOOL_VERSION,
        "experiment": "E2_variant3_floater_removal (LEDGER GOAL #008 pre-reg C)",
        "scene": args.scene,
        "source_ckpt": ckpt_fp,
        "source_model": src_model,
        "load_iteration": load_iter,
        "ft_iters": int(args.ft_iters),
        "floater_component_count": prune_payload["floater_component_count"],
        "removed_triangles": prune_payload["removed_triangles"],
        "removed_fraction": prune_payload["removed_fraction"],
        "n_triangles_source": prune_payload["n_triangles_source"],
        "n_triangles_final": prune_payload["audit"]["post_triangles"],
        "final_ckpt": final_ckpt,
        "ft_wallclock_min": (ft_payload or {}).get("ft_wallclock_min"),
        "train_command": shlex.join(train_cmd),
        "metrics_json": (eval_payload or {}).get("metrics_json"),
        "wandb_name": run_name,
    }
    row_path = os.path.join(out_model, "floater_prune_row.json")
    with open(row_path, "w") as f:
        json.dump(row, f, indent=1)
    print(f"[floater_prune] DONE {args.scene}\n[floater_prune] row: {row_path}",
          flush=True)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
