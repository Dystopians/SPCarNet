"""GEMS M2 — budget-engine pipeline (E1 engine, later the M6 backbone).

One command per (scene, budget, mode):
    source checkpoint -> [evidence] -> prune-to-budget -> [fine-tune]
    -> run_eval.py -> row.json

Usage:
    $PY -m tools.gems.gems_pipeline \
        --scene <garden|toy_parking|courtyard> \
        --source-ckpt <point_cloud_state_dict.pt> \
        --budget <0.5|0.25> \
        --mode <importance_ft|random_ft|importance_noft> \
        --tag <str> [--ft-iters 10000] [--gpu N] [--skip-eval]

Stages (each writes a stamp file keyed by its config hash under
<model_dir>/gems_stages/; re-running skips completed stages — D6
resumability):
  1 preflight  storage_preflight on /data/peilincai/gems_stage1/{models,eval}
  2 evidence   (importance modes) triangle_evidence npz for the SOURCE ckpt,
               cached per checkpoint fingerprint and reused across
               budgets/modes
  3 prune      keep_count = floor(budget * T_clean); importance modes keep the
               top-keep_count triangles by importance v1 = pixels_total
               (PRE-REGISTERED, untuned); random mode keeps a uniform subset
               of exactly keep_count (numpy default_rng(0)). Executed via
               ss3dm_prior/meshsplatopt/checkpoint_compaction.apply_compaction
               (keep-mask tensor surgery + vertex GC + index remap). Verified:
               pruned ckpt loads in TriangleModel, renders one TRAIN view,
               n_triangles == keep_count exactly.
  4 finetune   (ft modes) train.py called DIRECTLY, mirroring the strict
               compact-recovery recipe: topology frozen
               (--densify_until_iter=load_iter --skip_restricted_delaunay
               --freeze_topology_updates), --seed 0, --test_iterations -1
               (never fires; verified train.py membership test), wandb online.
               Post-FT topology audit asserts triangle/vertex counts
               unchanged.
  5 eval       run_eval.py (THE single mouth, D5)
  6 row        row.json with config hash, git commit, counts, wallclock,
               metrics path

D4 GUARDS: the evidence stage renders/reads TRAIN views only (asserted inside
tools/gems/triangle_evidence.py, test cameras are never image-loaded); the
fine-tune stage passes no test cameras beyond train.py's own --eval split
handling (--test_iterations -1 disables its test-metrics logging and
--wandb_disable_fixed_views its test-view image logging); this pipeline never
reads test GT itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable

GEMS_ROOT = "/data/peilincai/gems_stage1"
MODELS_ROOT = os.path.join(GEMS_ROOT, "models")
EVAL_ROOT = os.path.join(GEMS_ROOT, "eval")
EVIDENCE_ROOT = os.path.join(GEMS_ROOT, "evidence")

MODES = ("importance_ft", "random_ft", "importance_noft", "random_noft")
PIPELINE_VERSION = "gems_pipeline_v1"
RANDOM_PRUNE_SEED = 0  # numpy default_rng(0), pre-registered
TRAIN_SEED = 0


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def budget_label(budget: float) -> str:
    return f"B{int(round(budget * 100))}"


class StageStamps:
    """Per-stage stamp files under <model_dir>/gems_stages/. A stage is
    'done' iff its stamp exists AND the recorded config hash matches the
    current one (config change => stage re-runs; D6 resumability)."""

    def __init__(self, stamp_dir: str):
        self.stamp_dir = stamp_dir
        os.makedirs(stamp_dir, exist_ok=True)

    def path(self, stage: str) -> str:
        return os.path.join(self.stamp_dir, f"{stage}.stamp.json")

    def load(self, stage: str):
        p = self.path(stage)
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            return json.load(f)

    def is_done(self, stage: str, config_hash: str) -> bool:
        st = self.load(stage)
        return bool(st) and st.get("config_hash") == config_hash

    def write(self, stage: str, config_hash: str, payload: dict) -> None:
        rec = {
            "stage": stage,
            "config_hash": config_hash,
            "completed_utc": utc_now(),
            "git_commit": git_commit(),
            "payload": payload,
        }
        with open(self.path(stage), "w") as f:
            json.dump(rec, f, indent=1)


def _run_logged(cmd, log_path: str, env=None, cwd=REPO_ROOT) -> float:
    """Run a subprocess, tee stdout+stderr to log_path, raise on failure.
    Returns wall-clock seconds."""
    print(f"[gems] exec: {shlex.join(cmd)}\n[gems] log:  {log_path}", flush=True)
    t0 = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n===== {utc_now()} =====\n{shlex.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={proc.returncode}) after {dt:.0f}s: "
            f"{shlex.join(cmd)}\nsee {log_path}")
    return dt


def _ckpt_shapes(ckpt_path: str) -> dict:
    import torch
    state = torch.load(ckpt_path, map_location="cpu")
    out = {
        "triangles": int(state["_triangle_indices"].shape[0]),
        "vertices": int(state["triangles_points"].shape[0]),
    }
    del state
    return out


def _source_model_dir_and_iter(source_ckpt: str):
    """<model>/point_cloud/iteration_N/point_cloud_state_dict.pt -> (model, N)."""
    p = os.path.abspath(source_ckpt)
    assert os.path.isfile(p), f"source ckpt not found: {p}"
    assert os.path.basename(p) == "point_cloud_state_dict.pt", p
    iter_dir = os.path.dirname(p)
    pc_dir = os.path.dirname(iter_dir)
    model_dir = os.path.dirname(pc_dir)
    assert os.path.basename(pc_dir) == "point_cloud", (
        f"unexpected checkpoint layout (want <model>/point_cloud/iteration_N/): {p}")
    assert os.path.basename(iter_dir).startswith("iteration_"), p
    iteration = int(os.path.basename(iter_dir).split("_", 1)[1])
    return model_dir, iteration


# --------------------------------------------------------------------------
# stage implementations
# --------------------------------------------------------------------------

def stage_preflight(stamps: StageStamps, cfg_hash: str) -> dict:
    from tools.storage_preflight import volume_report
    reports = [volume_report(p, 50.0) for p in (MODELS_ROOT, EVAL_ROOT)]
    ok = all(r["ok"] for r in reports)
    if not ok:
        raise SystemExit(f"[gems] PREFLIGHT FAIL (D6): {json.dumps(reports)}")
    payload = {"volumes": reports}
    stamps.write("preflight", cfg_hash, payload)
    print(f"[gems] preflight OK: "
          f"{[(r['path'], r['free_gb']) for r in reports]}", flush=True)
    return payload


def evidence_npz_path(scene: str, ckpt_fp: dict) -> str:
    return os.path.join(
        EVIDENCE_ROOT,
        f"evidence_{scene}_{ckpt_fp['sha256_first16mb'][:12]}.npz")


def stage_evidence(scene: str, spec, source_ckpt: str, ckpt_fp: dict,
                   stamps: StageStamps, cfg_hash: str) -> dict:
    """Evidence npz for the SOURCE checkpoint, cached per checkpoint
    fingerprint (reused across budgets/modes). Returns the stamp payload
    (run_stage's cache-hit path returns the same dict shape)."""
    npz_path = evidence_npz_path(scene, ckpt_fp)
    meta_path = npz_path + ".meta.json"
    if os.path.isfile(npz_path) and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        cached_fp = (meta.get("checkpoint") or {}).get("sha256_first16mb")
        if cached_fp == ckpt_fp["sha256_first16mb"] and meta.get("max_views") is None:
            print(f"[gems] evidence cache HIT: {npz_path}", flush=True)
            payload = {"npz": npz_path, "cache": "hit", "meta": meta}
            stamps.write("evidence", cfg_hash, payload)
            return payload

    from tools.gems.triangle_evidence import compute_triangle_evidence
    meta = compute_triangle_evidence(
        checkpoint=source_ckpt, spec=spec, out_npz=npz_path, max_views=None)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    payload = {"npz": npz_path, "cache": "miss", "meta": meta}
    stamps.write("evidence", cfg_hash, payload)
    return payload


def _keep_ids(mode: str, keep_count: int, t_clean: int, npz_path=None):
    import numpy as np
    if mode.startswith("importance"):
        with np.load(npz_path) as z:
            pixels_total = z["pixels_total"]
        assert pixels_total.shape[0] == t_clean, (
            f"evidence npz has {pixels_total.shape[0]} triangles, "
            f"source ckpt has {t_clean}")
        # importance v1 = pixels_total (PRE-REGISTERED). Stable sort =>
        # deterministic tie-breaking by triangle id.
        order = np.argsort(-pixels_total.astype(np.int64), kind="stable")
        keep = np.sort(order[:keep_count])
        rule = "importance_v1_pixels_total_top_keep_count_stable"
    else:
        rng = np.random.default_rng(RANDOM_PRUNE_SEED)
        keep = np.sort(rng.choice(t_clean, size=keep_count, replace=False))
        rule = f"uniform_random_default_rng({RANDOM_PRUNE_SEED})"
    assert keep.shape[0] == keep_count
    return keep, rule


def stage_prune(scene: str, spec, source_ckpt: str, mode: str, budget: float,
                keep_count: int, t_clean: int, model_dir: str,
                npz_path, stamps: StageStamps, cfg_hash: str) -> dict:
    import numpy as np
    from ss3dm_prior.meshsplatopt.checkpoint_compaction import apply_compaction

    src_model, load_iter = _source_model_dir_and_iter(source_ckpt)
    keep, rule = _keep_ids(mode, keep_count, t_clean, npz_path)

    # apply_compaction takes the REMOVE list.
    keep_mask = np.zeros(t_clean, dtype=bool)
    keep_mask[keep] = True
    drop = np.nonzero(~keep_mask)[0]

    audit = apply_compaction(
        src_model, model_dir, load_iter, drop,
        selector_mode=f"gems_{mode}_{budget_label(budget)}",
    )
    assert audit.post_triangles == keep_count, (
        f"pruned triangle count {audit.post_triangles} != keep_count {keep_count}")
    assert audit.invalid_index_count == 0, "pruned ckpt has invalid face indices"

    pruned_ckpt = os.path.join(
        model_dir, "point_cloud", f"iteration_{load_iter}",
        "point_cloud_state_dict.pt")
    assert os.path.isfile(pruned_ckpt), pruned_ckpt

    # --- VERIFY: loads in TriangleModel and renders one TRAIN view. ---
    import torch
    from tools.gems.triangle_evidence import _TrainOnlyContext
    vctx = _TrainOnlyContext(pruned_ckpt, spec)
    n_loaded = int(vctx.faces().shape[0])
    assert n_loaded == keep_count, (
        f"loaded pruned ckpt has {n_loaded} triangles != keep_count {keep_count}")
    pkg = vctx.render_view(vctx.train_cams[0])
    img = pkg["render"].detach()
    assert img.dim() == 3 and img.shape[0] == 3 and torch.isfinite(img).all(), (
        "pruned checkpoint render sanity check failed")
    verify_view = vctx.train_cams[0].image_name
    del pkg, img, vctx
    torch.cuda.empty_cache()

    payload = {
        "pruned_ckpt": pruned_ckpt,
        "load_iteration": load_iter,
        "t_clean": t_clean,
        "keep_count": keep_count,
        "keep_rule": rule,
        "audit": audit.to_dict(),
        "verified_render_train_view": verify_view,
    }
    stamps.write("prune", cfg_hash, payload)
    print(f"[gems] prune OK: {t_clean} -> {keep_count} triangles "
          f"({rule})", flush=True)
    return payload


def build_train_cmd(spec, model_dir: str, load_iter: int, final_iter: int,
                    run_name: str, lr_overrides: dict | None = None) -> list:
    """Mirror scripts/car_model/meshsplatopt_run_strict_compact_recovery.py::
    _train_args (compact_render_only preset), calling train.py directly.

    lr_overrides: optional {'feature_lr': f, 'weight_lr': f,
    'lr_triangles_points_init': f} — E1 mechanism-variant knobs (PROTOCOL
    iteration budget applies); they enter the finetune stage hash via the
    command string."""
    cmd = [
        PY, "train.py",
        "-s", spec.source_path,
        "-m", model_dir,
        "--images", spec.images,
        "--resolution", str(spec.resolution),
        "--eval",
    ]
    if spec.split == "file5":
        split_file = os.path.join(spec.source_path, "split.json")
        assert os.path.isfile(split_file), f"split file missing: {split_file}"
        cmd += ["--split_strategy", "file", "--split_file", split_file]
    else:
        cmd += ["--split_strategy", "llff"]
    cmd += [
        "--load_iteration", str(load_iter),
        "--seed", str(TRAIN_SEED),
        "--iterations", str(final_iter),
        "--densify_until_iter", str(load_iter),
        "--skip_restricted_delaunay",
        "--freeze_topology_updates",
        # train.py's test-metric logging never fires at -1 (iteration is
        # always >= 1, membership test `iteration in [-1]` is never true);
        # this is the clean 'never' -- train.py is NOT patched.
        "--test_iterations", "-1",
        "--save_iterations", str(final_iter),
        # fixed-view wandb logging renders TEST cameras; disable via the
        # existing clean flag (D4 hygiene beyond train.py's own --eval split).
        "--wandb_disable_fixed_views",
        "--enable_wandb",
        "--wandb_project", "mesh-splatting",
        "--wandb_group", "gems_m2_budget_engine",
        "--wandb_name", run_name,
        "--wandb_image_log_interval", "1000",
        "--wandb_scalar_log_interval", "50",
    ]
    for k, v in (lr_overrides or {}).items():
        if v is not None:
            cmd += [f"--{k}", str(v)]
    return cmd


def stage_finetune(train_cmd: list, model_dir: str, pruned_ckpt: str,
                   final_iter: int, gpu, stamps: StageStamps,
                   cfg_hash: str) -> dict:
    # D4 guard: topology-frozen + no test-metric hooks in the exact command.
    for flag in ("--freeze_topology_updates", "--skip_restricted_delaunay",
                 "--wandb_disable_fixed_views"):
        assert flag in train_cmd, f"FT command lost required flag {flag}"
    ti = train_cmd.index("--test_iterations")
    assert train_cmd[ti + 1] == "-1", "FT must disable test-iteration logging"

    env = os.environ.copy()
    env["WANDB_MODE"] = "online"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    pre = _ckpt_shapes(pruned_ckpt)
    log_path = os.path.join(model_dir, f"gems_finetune_{final_iter}.log")
    dt = _run_logged(train_cmd, log_path, env=env)

    final_ckpt = os.path.join(
        model_dir, "point_cloud", f"iteration_{final_iter}",
        "point_cloud_state_dict.pt")
    assert os.path.isfile(final_ckpt), (
        f"fine-tune finished but final checkpoint missing: {final_ckpt}")

    # Post-FT topology audit: counts unchanged from the pruned ckpt.
    post = _ckpt_shapes(final_ckpt)
    assert post == pre, (
        f"topology changed during fine-tune: pruned={pre} final={post}")

    payload = {
        "final_ckpt": final_ckpt,
        "ft_wallclock_sec": dt,
        "ft_wallclock_min": dt / 60.0,
        "train_command": shlex.join(train_cmd),
        "log": log_path,
        "topology_audit": {"pruned": pre, "final": post, "unchanged": True},
    }
    stamps.write("finetune", cfg_hash, payload)
    print(f"[gems] finetune OK in {dt / 60.0:.1f} min; topology unchanged "
          f"{pre}", flush=True)
    return payload


def stage_eval(scene: str, final_ckpt: str, eval_dir: str, gpu,
               stamps: StageStamps, cfg_hash: str) -> dict:
    os.makedirs(eval_dir, exist_ok=True)
    cmd = [PY, "run_eval.py", "--checkpoint", final_ckpt,
           "--scene", scene, "--out", eval_dir]
    env = os.environ.copy()
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
        # run_eval sets CUDA_VISIBLE_DEVICES itself from --gpu; give it a
        # clean view of the machine so the index means the physical GPU.
        env.pop("CUDA_VISIBLE_DEVICES", None)
    log_path = os.path.join(eval_dir, "run_eval.log")
    dt = _run_logged(cmd, log_path, env=env)
    metrics_json = os.path.join(eval_dir, "metrics.json")
    assert os.path.isfile(metrics_json), f"run_eval wrote no metrics: {metrics_json}"
    payload = {"metrics_json": metrics_json, "eval_wallclock_sec": dt,
               "command": shlex.join(cmd)}
    stamps.write("eval", cfg_hash, payload)
    return payload


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GEMS M2 budget-engine pipeline")
    p.add_argument("--scene", required=True,
                   choices=None)  # validated against tools/gems/scenes.py SCENES in main()
    p.add_argument("--source-ckpt", required=True,
                   help="source point_cloud_state_dict.pt (clean baseline)")
    p.add_argument("--budget", type=float, required=True,
                   help="triangle budget as a fraction (0.5 or 0.25)")
    p.add_argument("--mode", required=True, choices=list(MODES))
    p.add_argument("--tag", required=True)
    p.add_argument("--ft-iters", type=int, default=10000)
    p.add_argument("--ft-feature-lr", type=float, default=None)
    p.add_argument("--ft-weight-lr", type=float, default=None)
    p.add_argument("--ft-position-lr", type=float, default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        # must happen before any torch import in this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    assert 0.0 < args.budget <= 1.0, f"budget must be in (0,1]: {args.budget}"

    from tools.gems.scenes import SCENES
    from tools.gems.triangle_evidence import checkpoint_fingerprint
    spec = SCENES[args.scene]

    blabel = budget_label(args.budget)
    name = f"{args.scene}_{blabel}_{args.mode}_{args.tag}"
    run_name = f"gems_{name}"
    model_dir = os.path.join(MODELS_ROOT, name)
    eval_dir = os.path.join(EVAL_ROOT, name)
    os.makedirs(model_dir, exist_ok=True)
    stamps = StageStamps(os.path.join(model_dir, "gems_stages"))

    source_ckpt = os.path.abspath(args.source_ckpt)
    _, load_iter = _source_model_dir_and_iter(source_ckpt)
    ckpt_fp = checkpoint_fingerprint(source_ckpt)
    t_clean = _ckpt_shapes(source_ckpt)["triangles"]
    keep_count = int(args.budget * t_clean)  # floor
    assert keep_count <= args.budget * t_clean  # hard budget (PROTOCOL section 2)
    final_iter = load_iter + int(args.ft_iters)
    with_ft = args.mode.endswith("_ft")
    with_evidence = args.mode.startswith("importance")

    final_ckpt = os.path.join(
        model_dir, "point_cloud",
        f"iteration_{final_iter if with_ft else load_iter}",
        "point_cloud_state_dict.pt")

    # ---- per-stage config strings -> hashes (stable, execution-detail-free)
    lr_overrides = {"feature_lr": args.ft_feature_lr,
                    "weight_lr": args.ft_weight_lr,
                    "lr_triangles_points_init": args.ft_position_lr}
    train_cmd = build_train_cmd(spec, model_dir, load_iter, final_iter,
                                run_name, lr_overrides) if with_ft else []
    stage_cfg = {
        "preflight": f"{PIPELINE_VERSION}|preflight|roots={MODELS_ROOT},{EVAL_ROOT}|min_free_gb=50",
        "evidence": (f"{PIPELINE_VERSION}|evidence|scene={args.scene}"
                     f"|ckpt={ckpt_fp['sha256_first16mb']}"
                     f"|bytes={ckpt_fp['file_size_bytes']}|max_views=None"),
        "prune": (f"{PIPELINE_VERSION}|prune|scene={args.scene}"
                  f"|ckpt={ckpt_fp['sha256_first16mb']}|budget={args.budget}"
                  f"|mode={args.mode}|t_clean={t_clean}|keep_count={keep_count}"
                  f"|importance=pixels_total_v1|random_seed={RANDOM_PRUNE_SEED}"),
        "finetune": (f"{PIPELINE_VERSION}|finetune|{shlex.join(train_cmd)}"
                     if with_ft else ""),
        "eval": (f"{PIPELINE_VERSION}|eval|scene={args.scene}"
                 f"|checkpoint={final_ckpt}|out={eval_dir}"),
    }
    stage_hash = {k: sha256_str(v) for k, v in stage_cfg.items()}
    # row config hash = sha256 of the FULL stage-config string
    full_cfg_str = "\n".join(
        f"{k}::{stage_cfg[k]}" for k in
        ("preflight", "evidence", "prune", "finetune", "eval"))
    config_hash = sha256_str(full_cfg_str)

    print(f"[gems] run={name} T_clean={t_clean} keep_count={keep_count} "
          f"config_hash={config_hash[:12]}", flush=True)

    stage_wallclock = {}

    def run_stage(stage_name, fn, *fn_args, **fn_kwargs):
        h = stage_hash[stage_name]
        if stamps.is_done(stage_name, h):
            print(f"[gems] stage '{stage_name}' already complete "
                  f"(hash match) — skipping", flush=True)
            return stamps.load(stage_name)["payload"]
        t0 = time.time()
        payload = fn(*fn_args, **fn_kwargs)
        stage_wallclock[stage_name] = time.time() - t0
        return payload

    # (1) preflight — always evaluated fresh (disk state is external)
    t0 = time.time()
    stage_preflight(stamps, stage_hash["preflight"])
    stage_wallclock["preflight"] = time.time() - t0

    # (2) evidence (importance modes only; cached per ckpt fingerprint)
    npz_path = None
    if with_evidence:
        payload = run_stage("evidence", stage_evidence, args.scene, spec,
                            source_ckpt, ckpt_fp, stamps,
                            stage_hash["evidence"])
        npz_path = payload["npz"]

    # (3) prune
    prune_payload = run_stage(
        "prune", stage_prune, args.scene, spec, source_ckpt, args.mode,
        args.budget, keep_count, t_clean, model_dir, npz_path, stamps,
        stage_hash["prune"])
    pruned_ckpt = prune_payload["pruned_ckpt"]

    # (4) fine-tune (ft modes)
    ft_payload = None
    if with_ft:
        ft_payload = run_stage(
            "finetune", stage_finetune, train_cmd, model_dir, pruned_ckpt,
            final_iter, args.gpu, stamps, stage_hash["finetune"])

    assert os.path.isfile(final_ckpt), f"final checkpoint missing: {final_ckpt}"

    # (5) eval
    eval_payload = None
    if args.skip_eval:
        print("[gems] --skip-eval: stage 5 skipped", flush=True)
    else:
        eval_payload = run_stage("eval", stage_eval, args.scene, final_ckpt,
                                 eval_dir, args.gpu, stamps,
                                 stage_hash["eval"])

    # (6) row
    row = {
        "scene": args.scene,
        "budget": args.budget,
        "budget_label": blabel,
        "mode": args.mode,
        "tag": args.tag,
        "config_hash": config_hash,
        "git_commit": git_commit(),
        "source_ckpt": ckpt_fp,
        "n_triangles_clean": t_clean,
        "n_triangles_pruned": prune_payload["keep_count"],
        "final_ckpt": final_ckpt,
        "ft_iters": int(args.ft_iters) if with_ft else 0,
        "ft_wallclock_min": (ft_payload or {}).get("ft_wallclock_min"),
        "metrics_json": (eval_payload or {}).get("metrics_json"),
        "eval_skipped": bool(args.skip_eval),
        "stage_wallclock_sec": stage_wallclock,
        "wandb_name": run_name if with_ft else None,
        "written_utc": utc_now(),
    }
    os.makedirs(eval_dir, exist_ok=True)
    row_path = os.path.join(eval_dir, "row.json")
    with open(row_path, "w") as f:
        json.dump(row, f, indent=1)
    stamps.write("row", sha256_str(config_hash), {"row_json": row_path})
    print(f"[gems] DONE {name}\n[gems] row: {row_path}", flush=True)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
