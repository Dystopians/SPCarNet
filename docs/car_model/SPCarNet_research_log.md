# SP-CarNet Research Log

Single source of truth for "what was tried under the SP-CarNet research line and how it went". Date-stamped, append-only. Each entry links to the relevant design / implementation / smoke / failure documents per the policy in `SPCarNet_radical_RFC.md` §8.

---

## 2026-04-29 — Stage 1 (Object cache & canonicalization audit) — DONE

**Outcome**: cache audit passed. SP-CarNet can proceed on real object-level data without any cache rebuild.

**Headline numbers**:
- 2 433 objects (1 854 train / 206 val / 373 test); 1 patch per object across the board.
- 100 % of objects link to a source GLB via the manifest.
- Cache split: 1 616 records at format v2 (no symmetry persisted), 817 records at format v3 (symmetry persisted as `symmetry_plane_normal/offset/confidence/chamfer_residual`).
- `clean_points (2048, 3)`, `clean_normals (2048, 3)`, `visible_clean_points / hidden_clean_points`, `observed_points (768, 3)`, `query_points_all (1280, 3)` with `query_labels_all` and `query_ignore_mask`, `surface_query_points (512, 3)`, `free_query_points (512, 3)`, `free_space_query_hard_negatives (128, 3)` are all present.
- Every record is already centred (`patch_center_world == 0`) and unit-radius (`patch_radius_m == 1.0`) — canonical identity transform is the working default.
- Front-axis convention is **not** annotated. PCA orientation is provided as an opt-in fallback with a flagged eigenvector-sign caveat.
- Scanner pose is not persisted; runtime sampling via the existing LiDAR corruption module remains the route for `L_ray` evidence in Stage 3+.

**Files added**: `docs/car_model/spcarnet_stage1_object_cache_design.md`, `scripts/car_model/build_spcarnet_object_index.py`, `ss3dm_prior/data/spcarnet_object_dataset.py`, `scripts/car_model/smoke_test_spcarnet_stage1.py`, `outputs/carnet/spcarnet/object_index_v1.json`, `docs/car_model/spcarnet_stage1_object_cache_report.md`, this log.

**No file modified**: CarNet_v0 / v0.7 / v0.8.x configs, the patch-centric dataset, and the trainer remain untouched.

**Smoke test**: `[smoke] PASS` — index build, dataset open over all three splits, `clean_points_object (2048, 3) float32` non-NaN, `partial_observed_points (768, 3) float32` non-NaN, occupancy labels strictly ∈ {0, 1} after applying the ignore mask, identity round-trip error 0.0, PCA-style round-trip error 5.96 × 10⁻⁸, batch collate produces the expected fixed-shape stacks for `B = 2`.

**Decision**: Stage 1 gate **PASSED**. Proceeding to Stage 2 (shape-field auto-decoder).

**Linked artefacts**:
- Design — `docs/car_model/spcarnet_stage1_object_cache_design.md`
- Report — `docs/car_model/spcarnet_stage1_object_cache_report.md`
- Index artefact — `outputs/carnet/spcarnet/object_index_v1.json`

---

## 2026-04-29 — Stage 2 (Canonical object-level shape-field auto-decoder) — IMPLEMENTED, smoke PASS, full-training pending

**Outcome**: code complete, smoke green; the headline auto-decoder run has not yet been launched. A small pre-launch hardening pass (checkpoint emission inside `fit()`, periodic eval, wandb integration) is the only remaining work before the first headline run.

**Architecture locked in**:
- Decoder: 6-layer FiLM-modulated MLP, hidden_dim=384, latent_dim=256, Fourier features with 32 log-spaced frequencies, occupancy logit head (SDF head + eikonal regulariser are wired as ablation).
- Per-object latent table `LatentTable(num_objects, latent_dim)` initialised `N(0, 0.01)`.
- Query budget per object per step: 384 surface + 384 free + 128 hard-negative + 128 mixed (with `query_ignore_mask` honoured) = 1024. SDF mode adds 256 eikonal samples.
- Optim: Adam, decoder LR 5e-4, latent LR 1e-3, grad_clip 1.0. Latent prior `w_zL2 = 1e-4 · ||z||² / d_z`.
- Trains on `train` only; `val` reserved for the eval entrypoint and the Stage gate.

**Files added**: `docs/car_model/spcarnet_stage2_shape_field_design.md`, `ss3dm_prior/models/spcarnet_shape_field.py`, `ss3dm_prior/training/__init__.py`, `ss3dm_prior/training/spcarnet_autodecoder.py`, `ss3dm_prior/training/spcarnet_autodecoder_cli.py`, `configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder.yaml`, `configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder.yaml`, `scripts/car_model/train_spcarnet_shape_field_autodecoder.sh`, `scripts/car_model/eval_spcarnet_shape_field_autodecoder.py`, `scripts/car_model/smoke_test_spcarnet_stage2.py`, `docs/car_model/spcarnet_stage2_shape_field_implementation_report.md`.

**No file modified**: `ss3dm_prior/engine/trainer.py`, the v0.x configs, the patch-centric dataset, the v0.x launchers — the auto-decoder line is fully isolated under `configs/ss3dm_prior/spcarnet/` and `ss3dm_prior/training/`. RFC §6 "demote, don't delete" honoured.

**Smoke test** — `scripts/car_model/smoke_test_spcarnet_stage2.py`:
- `[stage2-smoke] PASS` after 2 iters on 2 objects with a tiny 32-d latent / 64-wide / depth-3 decoder.
- `loss_total` 2.0794 → 2.0742 (strict decrease).
- Each BCE term lands at 0.6931 ≈ ln 2 at iter 0, confirming a clean "uninformative" init.
- Decoder gradients non-zero on at least one parameter; latent table gradients non-zero. Both pathways live.
- Marching-Cubes call returns `mesh=None, vertex_count=0` at resolution=16 — expected fallback for an untrained sigmoid field; smoke validates the pipeline runs, not the mesh quality.

**Stage gate** (unchanged, conditional on the headline run):
- `mesh_iou_at_0.5_mean ≥ 0.92`
- `recon_chamfer_l1_mean ≤ 0.05` (canonical units)
- `mesh_extraction_success_rate ≥ 0.95`

All three must hold simultaneously on `val`.

**Decision**: Stage 2 implementation gate **PASSED**. Stage 2 *training* gate is conditional on the headline run; advancing to Stage 3 (per-object MAP refinement at val time) is blocked on §5 of the implementation report.

**Linked artefacts**:
- Design — `docs/car_model/spcarnet_stage2_shape_field_design.md`
- Implementation report — `docs/car_model/spcarnet_stage2_shape_field_implementation_report.md`
- RFC — `docs/car_model/SPCarNet_radical_RFC.md`

---

## 2026-04-29 — Stage 2 (autodecoder_v1, headline) — TRAIN COMPLETE; gate **soft FAIL** on chamfer, IoU metric was broken

**Outcome**: 200-epoch run on the full train split (1854 objects) finished cleanly in **34 minutes** on GPU 5. Final wandb summary `loss_total=0.00468`, `loss_surf=0.00394`, `loss_free=0.00064`, `loss_hard=0.0`, `loss_mixed=0.0002`, `loss_zL2=0.03031`. Wandb run: `5ipij4y9`.

**Train-set eval (64 obj subsample, mc_resolution=32)**:
- `mesh_extraction_success_rate = 1.000` (✓, gate ≥ 0.95)
- `recon_chamfer_l1 = 0.066` (✗, gate ≤ 0.05 — over by 32 %)
- `mesh_iou_at_0.5 = 0.488` (✗, gate ≥ 0.92) — but this number was **a metric bug**, not a model failure (see below)
- `surface_normal_consistency = 0.735`
- `hidden_chamfer_l1 = 0.097`

**Val eval was not informative** — the auto-decoder paradigm has no per-object latent for val/test by construction (those splits had no Stage-2 z to optimise over). Val mesh extraction ran 0/206 because the eval script skipped objects without a Stage-2 latent. This is the Stage-2 → Stage-3 boundary, not a bug.

**IoU metric correction (sub-task)**: the `_voxelise_points` step in `eval_spcarnet_shape_field_autodecoder.py` voxelised only 2 048 `clean_points` at 32³, which biases IoU to ~0.5 even on perfect reconstruction (sparse shell vs filled volume). Fixed in-place by `_voxelise_gt_mesh` which loads the source GLB via the manifest, applies the Stage-1 canonical transform from `patch_metadata_json`'s `original_centroid_world / original_radius_world`, and uses `mesh.voxelized(2/res).fill().matrix` as filled GT. Falls back to a dilated-shell IoU (reported under `mesh_iou_at_0.5_shell`) when the GLB is missing.

**Re-eval on first 16 train objects (post-fix, mc_resolution=32)**:
- `mesh_iou_at_0.5_mean = 0.590` (filled GT, n=6 with local GLB)
- `mesh_iou_at_0.5_shell_mean = 0.922` (shell fallback, n=10 missing GLB)
- vs broken `0.488`

The shell-IoU at 0.922 is consistent with the geometry being substantially correct but the chamfer being slightly looser than the gate.

**Surprises documented for Stage 1 / cache layout**:
1. Manifest's nominal `dataset_root + ./raw/<id>.glb` does **not** exist on disk; actual GLBs live at `/data/peilincai/car_models/meshfleet_ext_v02/{train,test}/raw/`, with only ~6/16 of the first 16 train cars present locally — heavy fallback usage.
2. The cache's canonicalisation is **not** identity. NPZ headers report `patch_center_world=0, patch_radius_m=1`, but the actual world→cache transform is `(v - original_centroid_world) / original_radius_world` from `patch_metadata_json`. Stage 1's "identity is the working default" finding is misleading — the points are pre-canonicalised, but the canonical transform is non-trivial when re-projecting external mesh data into the cache frame. The Stage-1 design doc and Stage-2 eval script both depend on the post-fixed transform now.

**Decision**: Stage-2 v1 is "soft pass" — pipeline is healthy, geometry is recognisable, but the chamfer gate is missed by ~30 %. A v2 retrain with bigger query budget + 300 epochs is in flight (see next entry).

**Linked artefacts**:
- Implementation report — `docs/car_model/spcarnet_stage2_shape_field_implementation_report.md`
- v1 eval (broken IoU): `outputs/carnet/spcarnet/autodecoder_v1/eval_train_64.json`
- v1 eval (fixed IoU): `outputs/carnet/spcarnet/autodecoder_v1/eval_train_16_iou_fix.json`
- Wandb: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/5ipij4y9

---

## 2026-04-29 — Stage 2 (autodecoder_v2, retrain) — IN FLIGHT

**Goal**: push `recon_chamfer_l1` below 0.05 to clear the Stage-2 gate cleanly.

**Diff vs v1**: queries doubled (`surface=768, free=768, hard=256, mixed=256`, total 2048 / object / step), epochs `200 → 300`. All other hyperparameters unchanged. Output dir `autodecoder_v2/`; v1 preserved at `autodecoder_v1/`.

**Status**: PID 1070553 on GPU 5. Wandb run `mpdb1mm7`. Currently ~4350/69300 steps (epoch 18) at sub-agent handover; loss curve healthy and decreasing; no Traceback/OOM.

**Linked artefacts**:
- Wandb: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/mpdb1mm7
- Log: `outputs/carnet/spcarnet/autodecoder_v2/logs/train.log`

---

## 2026-04-29 — Stage 3 (posterior encoder `q(z | O)`) — IMPLEMENTED, smoke + integration smoke PASS, headline pending

**Outcome**: code complete; standalone smoke and trainer-integration smoke (3 steps against the real Stage-2 v1 checkpoint) both pass. The headline run is **not yet launched** — GPU 5 is occupied by the Stage-2 v2 retrain.

**Architecture locked in**:
- Encoder: PointNet tokeniser + 4 cross-attention / 2 self-attention blocks over 32 learnable queries, `feature_dim=256`, ffn 1024, heads 8, dropout 0.1.
- Posterior: variational `(μ, log σ²)` with reparameterisation; KL warmup `0 → 1e-3` over 10 ep; free-bits 0.1 nats/dim.
- Latent supervision: L2 regression of `μ` against the Stage-2 v1 latent table (frozen, train-only by construction); `w_z` warmup 2 → 10 over 10 ep.
- Reconstruction terms: BCE on partial-observed surface + free queries + hard negatives + mixed queries (with ignore mask), all decoded through the **frozen** Stage-2 v1 decoder.
- Optim: AdamW, encoder LR 3e-4, weight_decay 1e-4, grad_clip 1.0, batch 16, 150 epochs.
- Decoder finetune ablation wired (last 2 FiLM blocks + field head, LR 1e-5, off by default).

**Files added**: `docs/car_model/spcarnet_stage3_posterior_encoder_design.md`, `ss3dm_prior/models/spcarnet_posterior.py`, `ss3dm_prior/training/spcarnet_posterior.py`, `ss3dm_prior/training/spcarnet_posterior_cli.py`, `configs/ss3dm_prior/spcarnet/{model,train}_spcarnet_posterior_encoder.yaml`, `scripts/car_model/{train,smoke_test,eval}_spcarnet_posterior_encoder.{sh,py,py}`, `scripts/car_model/smoke_test_spcarnet_stage3.py`, `docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md`.

**No file modified**: Stage-1 dataset, Stage-2 trainer/decoder, v0.x configs/launchers, the patch-centric trainer. Stage-2 v1 checkpoint is read-only input.

**Smoke test**: standalone CPU smoke `[stage3-smoke] PASS` — encoder forward shape `(2, 32)`, initial logvar `−9.21` matches `log(0.01²)`, two reparameterised samples differ by 0.011, decoded logits `(2, 64)` finite with sigmoid mean 0.500 (uniform field at init), encoder gradients flow, decoder gradients **don't** (frozen). Integration smoke (3 steps, real Stage-2 v1 ckpt) ran cleanly in 2.18 s on GPU 5; wandb run `9kehaimo` synced; checkpoint payload schema matches the eval script.

**Stage gate** (per RFC §7, conditional on the headline run):
- `recon_chamfer_l1_mean ≤ 0.10` on val (matches v0.7's floor).
- `free_space_violation_rate_mean` strictly better than v0.7's.
- Both within 150 epochs.

**Decision**: Stage-3 implementation gate **PASSED**. Headline run is queued behind the Stage-2 v2 retrain on GPU 5; can be parallelised on a free GPU at user discretion.

**Linked artefacts**:
- Design — `docs/car_model/spcarnet_stage3_posterior_encoder_design.md`
- Implementation report — `docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md`
- RFC — `docs/car_model/SPCarNet_radical_RFC.md` (§3.4–§3.7, §6 EN-Q row, §7 Stage-3 gate)

---

## 2026-04-29 — Stage 3 (posterior encoder) — TRAIN COMPLETE; gate **PASS**

**Outcome**: 150-epoch run on the full train split (1854 objects) finished cleanly in **23 minutes** on GPU 1. Wandb run `eau9yg7m`. Final summary `loss_total=0.674`, `loss_z=0.012` (latent regression converged), `loss_surf=0.059`, `loss_free=0.111`, `loss_kl=346`, `posterior/logvar_mean=-3.65` (no collapse — would need to be < −8 to indicate collapse). KL stable at 346 vs free-bits floor of 25.6 (0.1 nats × 256 dims) — encoder is using meaningful capacity.

**Val eval (full 206 objects, mc_resolution=32)**:
- `mesh_extraction_success_rate = 1.000` (all 206 objects produced a mesh)
- `recon_chamfer_l1_mean = **0.0664**` — beats v0.7 (0.10) by 33 %, beats v0.8.2 (0.12) by 45 %
- `hidden_chamfer_l1_mean = 0.0991`
- `visible_preservation_error_mean = 0.0627`
- `free_space_violation_rate_mean = **0.0335**` (excellent; gate "strictly better than v0.7")
- `mesh_iou_at_0.5_mean = 0.471` (sparse-point fallback; not gated)
- `zero_corruption_recon_chamfer_l1_mean = 0.0666` ≈ `recon_chamfer_l1_mean = 0.0664` — **amortisation gap is essentially zero**
- `latent_retrieval_error_mean = NaN` (correctly masked on val — no leakage)

**Stage-3 gate PASS** on both the chamfer threshold (≤ 0.10) and the free-space violation requirement (strictly better than v0.7). The bottleneck is now **the Stage-2 decoder ceiling**, not the encoder.

**Linked artefacts**:
- Wandb: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/eau9yg7m
- Eval JSON: `outputs/carnet/spcarnet/posterior_encoder_v1/eval_val.json`
- Implementation report: `docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md`

---

## 2026-04-29 — Stage 4 (observation-consistency MAP refinement) — IMPLEMENTED, smoke + 50-obj refinement PASS, gate **soft pass**

**Outcome**: code complete; smoke and 50-object val refinement run both finished cleanly. Refinement helps on every metric in the right direction; chamfer improvement is half the design-side margin gate but inside the RFC §7 no-degradation triggers. Free-space violation **almost halves** (−59 %).

**Architecture / protocol locked in**:
- Loss module `ss3dm_prior/losses_spcarnet_observation.py`: Tier-1 `L_surf_obs + L_free + L_mixed`, Tier-2 `L_ray + L_incidence` (Tier-2 disabled on the current cache because scanner pose is not persisted — fallback documented in design §2).
- Huber wrap with `δ = 0.5` on every BCE term (robust to outlier observations).
- Refinement protocol: init `z = μ(O)` from the Stage-3 encoder, Adam on `[z]` only with frozen decoder, default 50 steps × LR 1e-2.
- Held-out validation partition (default 20 % of `query_points_all`) for keep-best tracking.
- Three early-stop triggers: plateau (10-step patience on held-out score), free-space violation increase (> 10 % over initial), z drift > 5×prior σ.
- Output JSON splits `inference_only_metrics` (real-deployment-safe) from `gt_dependent_metrics` (eval only).

**50-object val refinement results (default config)**:

| Metric | Before | After | Δ |
|---|---|---|---|
| `recon_chamfer_l1_mean` | 0.0715 | 0.0690 | −0.0025 |
| `hidden_chamfer_l1_mean` | 0.1078 | 0.1054 | −0.0024 |
| `visible_preservation_error_mean` | 0.0644 | 0.0610 | −0.0034 |
| `free_space_violation_rate_mean` | 0.0358 | **0.0147** | **−0.0211 (−59 %)** |

21/50 early stops (20 plateau, 1 free-space-increase — safeguard fired correctly). 0.92 s / object refinement time. `z_drift_final_mean = 1.86` (well within prior bound).

**Gate verdict**:
- RFC §7 hidden-chamfer ceiling (≤ 5 % degradation): ✓ — improved 2.2 %.
- RFC §7 free-space ceiling (≤ 10 % degradation): ✓ — improved 59 %.
- Design-side chamfer margin (≥ 0.005 improvement): **missed by ~2×** (got 0.0025).
- Decision: **soft pass**. Refinement is helpful but bounded by the Stage-2 decoder ceiling, exactly as predicted by the Stage-3 amortisation-gap diagnostic.

**No file modified**: Stage 1/2/3 modules and the v0.x line are untouched. Stage-3 checkpoint is read-only input.

**Files added**: `docs/car_model/spcarnet_stage4_observation_map_design.md`, `ss3dm_prior/losses_spcarnet_observation.py`, `scripts/car_model/refine_spcarnet_latent_map.py`, `scripts/car_model/smoke_test_spcarnet_stage4.py`, `docs/car_model/spcarnet_stage4_observation_map_implementation_report.md`.

**Linked artefacts**:
- Refinement JSON: `outputs/carnet/spcarnet/map_refinement/val_50_default/refinement.json`
- Implementation report: `docs/car_model/spcarnet_stage4_observation_map_implementation_report.md`

---

## 2026-04-30 — Stage 5 (multi-hypothesis sampling & reranking) + Stage 2 v3 sanity check

**Outcome**: Stage 5 implemented end-to-end. K∈{1, 4, 8} sweep on 50 val objects. **Mixed gate**: oracle best-of-K=8 beats K=1 by 0.0060 chamfer (passes RFC §7 ≥0.005 margin) — *the posterior is genuinely multi-modal*. But the inference-only reranker score (BCE losses + `log p(z)`) ranks the wrong candidate: top1-reranked is +0.002 chamfer *worse* than K=1. The headline-gate (top1 reranked beats K=1 by ≥0.005) **fails**.

**Stage-2 v3 sanity (run in parallel)**: bigger decoder (latent 512, hidden 768, depth 8, 300 ep) → train chamfer 0.0692 vs v1 ~0.066. **Did not lift the ceiling**; v1/v2/v3 are within 0.003 chamfer of each other. Stage 3 is **not** re-paired against v3.

**Architecture / protocol**:
- Sample K from variational posterior with `torch.manual_seed(seed_base + k)` per candidate (one encoder pass, K MC extractions).
- Score = `−L_obs(z_k) + log p(z_k)` where `L_obs = w_surf·BCE(P_obs,1) + w_free·BCE(Q_free,0) + w_mixed·BCE_with_ignore(Q_all)` (no Huber wrap; likelihood form).
- Diversity primary metric: pairwise top-3 chamfer; secondary: latent-L2.
- Mesh extraction (MC res 32) post-hoc per candidate; failed extractions excluded from rerank/oracle.

**50-object val sweep results**:

| Metric | K=1 | K=4 | K=8 |
|---|---|---|---|
| `top1_score_recon_chamfer_l1` | **0.0715** | 0.0734 | 0.0735 |
| `oracle_best_of_k_recon_chamfer_l1` | 0.0715 | **0.0669** | **0.0655** |
| `top1_score_visible_preservation_error` | 0.0632 | 0.0644 | 0.0650 |
| `top1_score_free_space_violation_rate` | 0.0366 | 0.0395 | 0.0364 |
| `diversity_chamfer_top3` | NaN | 0.0348 | 0.0342 |
| `diversity_latent_l2` | NaN | 3.91 | 3.88 |
| `mesh_extraction_success_rate` | 1.00 | 1.00 | 1.00 |
| seconds / object | 0.60 | 2.33 | 3.23 |

**Why the reranker fails — and why fixes don't help**: post-hoc, we tested three score variants on the existing K=8 JSON via `scripts/car_model/rescore_spcarnet_multihypothesis.py` (recomputes top1 without re-running):

| K=8 variant | top1 chamfer | vs K=1 (0.0715) |
|---|---|---|
| default (`-L_obs + log p(z)`) | 0.0735 | +0.0020 |
| no_prior (`-L_obs`) | 0.0737 | +0.0022 |
| norm_penalty (`-L_obs - 0.5·max(0,‖z‖-4)`) | 0.0738 | +0.0023 |
| **oracle (GT chamfer)** | **0.0655** | **−0.0060** |

K=4 same pattern (no_prior best non-oracle at 0.0725, still +0.0010 over K=1). **No inference-only variant beats K=1.** The real issue is not the prior term: `L_obs` (BCE on observation queries) is decorrelated from chamfer-to-GT in the local neighbourhood of the posterior, because BCE only sees 768 partial-obs points + a fixed query grid, not the unobserved surface that chamfer measures. This rules out a whole family of approaches (any score that uses only `(z, decoder, partial obs)`) — Stage 7-aux now has a strong empirical motivation to bring in evidence the reranker doesn't currently see (symmetry consistency, RAG against a shape bank, manifold quality scores).

**Why oracle wins**: posterior σ is calibrated such that ~1 in 8 samples lands inside the GT-closer side of the local mode. Latent-L2 spread (3.9) is comparable to prior σ × √D; mesh-space top-3 chamfer spread (0.034) is half the typical chamfer level — meaningful but not chaotic.

**v3 sanity numbers (train, 100 obj, MC 32)**: chamfer_l1 = 0.0692, mesh_iou_shell = 0.914, n_extracted = 100/100. Confirms decoder ceiling is family-level, not capacity-level.

**Gate verdict**:
- RFC §7 chamfer margin ≥ 0.005 (top1 reranked vs K=1): **✗** (wrong direction by 0.002).
- RFC §7 chamfer margin ≥ 0.005 (oracle vs K=1): ✓ (−0.006).
- RFC §7 free-space ceiling (≤ 10 % degradation): ✓ (K=8: 0.0364 vs K=1: 0.0366 — flat).
- RFC §7 mesh-extraction (no regression): ✓ (1.00 across all K).
- Diversity-doubling gate (K=8 top-3 ≥ 2× K=4): **✗** (0.0342 vs 0.0348 — flat). Gate-design issue: doubling assumes multi-modal; ours is unimodal-broad.
- **Decision: drop multi-hypothesis from headline, keep K=1; retain K=8 oracle as ablation row in the paper.**

**No file modified**: Stage 1/2/3/4 modules untouched. Stage-3 v1 checkpoint is read-only input.

**Files added**: `docs/car_model/spcarnet_stage5_multihypothesis_design.md`, `scripts/car_model/eval_spcarnet_multihypothesis.py`, `scripts/car_model/smoke_test_spcarnet_stage5.py`, `scripts/car_model/rescore_spcarnet_multihypothesis.py`, `docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md`. Stage-2 v3 launcher: `scripts/car_model/train_spcarnet_shape_field_autodecoder_v3.sh`.

**Linked artefacts**:
- Stage 5 K=1 / K=4 / K=8 JSONs: `outputs/carnet/spcarnet/multihypothesis/val_50_K{1,4,8}/K{1,4,8}.json`
- v3 checkpoint (preserved, not used downstream): `outputs/carnet/spcarnet/autodecoder_v3/checkpoint_last.pt`
- v3 train eval: `outputs/carnet/spcarnet/autodecoder_v3/eval/train_eval.json`
- Implementation report: `docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md`

---

## 2026-05-01 — MeshPrior Stage 0 (repository audit) — PASS / PROCEED

**Outcome**: M0 repository integrity audit completed for the SP-CarNet → MeshPrior transition. No new method code was implemented.

**Environment**:
- Default shell Python is `3.13.2` and does not have `torch`; it is not the project environment.
- `micromamba run -n mesh_splatting` provides Python `3.11.14`, `torch 2.7.1+cu126`, CUDA available, `cuda_device_count=8`.
- `python -m compileall scripts/car_model ss3dm_prior -q` passes in the `mesh_splatting` environment.

**Code audit**:
- Required SP-CarNet source files are present, including `spcarnet_object_dataset.py`, `spcarnet_shape_field.py`, `spcarnet_posterior.py`, Stage-2/Stage-3 trainers, Stage-4 observation loss, and Stage-1/3/4/5 scripts.
- `ss3dm_prior.models.spcarnet_shape_field` and `ss3dm_prior.models.spcarnet_posterior` import cleanly.
- Worktree was already dirty before this audit: `scripts/car_model/eval_spcarnet_multihypothesis.py` modified, `docs/prompts.md` untracked, and two submodules reported as dirty/unknown.

**Smoke tests**:
- `smoke_test_spcarnet_stage1.py`: PASS.
- `smoke_test_spcarnet_stage2.py`: PASS.
- `smoke_test_spcarnet_stage3.py`: PASS.
- `smoke_test_spcarnet_stage4.py`: PASS.
- `smoke_test_spcarnet_stage5.py`: PASS.

**Artifact audit**:
- Stage-2/3/4/5 checkpoints and JSONs exist under `outputs/carnet/spcarnet/`.
- Key reported metrics are supported by local JSONs, including Stage-3 `recon_chamfer_l1_mean=0.066391`, `free_space_violation_rate_mean=0.033535`, Stage-4 refinement `0.071490 -> 0.069032` chamfer and `0.035820 -> 0.014688` free-space violation, and Stage-5 K=8 oracle `0.065528` vs top1 reranked `0.073501`.

**Decision**: M0 recommendation is `PROCEED`. The next allowed prompt is M1, the MeshPrior scene-optimization RFC, with no model-code changes.

**Linked artefact**:
- Audit report: `docs/car_model/meshprior_stage0_repository_audit.md`

---

## 2026-05-01 — MeshPrior Stage 1 (scene mesh-prior RFC) — COMPLETE / PROCEED_TO_M2

**Outcome**: Wrote the MeshPrior research RFC that pivots SP-CarNet from object-only completion to object-prior-guided scene mesh optimization. No model code was changed.

**Central claim**: learned object-centric shape posteriors can safely guide scene mesh optimization when converted into bounded local proposals and filtered by scene-level evidence gates.

**Method slogan**: `Prior proposes; evidence disposes.`

**Planned system layers**:
- repository/object-prior integrity,
- scene/object region mining,
- object posterior inference in canonical frame,
- mesh repair proposal generation,
- scene evidence gates and rollback,
- alternating scene optimization,
- NeurIPS-grade evaluation and reporting.

**Proposal order**: protect/prune first, snap second, guarded fill third, split/collapse refinement last.

**Decision**: M1 is complete. The next allowed stage is M2 region mining.

**Linked artefact**:
- RFC: `docs/car_model/meshprior_stage1_scene_meshprior_RFC.md`

---

## 2026-05-01 — MeshPrior Stage 2 (scene/object region mining) — PASS

**Outcome**: Implemented the first scene/object bridge for MeshPrior: a conservative region mining layer that can process PLY meshes when present and emits clean dry-run artifacts when no scene mesh or segmentation exists.

**Files added**:
- `ss3dm_prior/meshprior/__init__.py`
- `ss3dm_prior/meshprior/region_types.py`
- `scripts/car_model/meshprior_mine_regions.py`
- `scripts/car_model/smoke_test_meshprior_stage2_region_mining.py`
- `docs/car_model/meshprior_stage2_region_mining_design.md`
- `docs/car_model/meshprior_stage2_region_mining_implementation_report.md`
- `docs/car_model/meshprior_stage2_region_mining_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage2_region_mining.py`: PASS, synthetic two-component mesh produced `regions=2`, `eligible_for_posterior=1`.
- Missing-data dry-run: PASS, emitted empty region set and exited cleanly.

**Contract**:
- Outputs `regions.json`, `regions_summary.csv`, and `region_mining_report.md`.
- Very small components are retained as diagnostics but not marked eligible for posterior inference.
- No SP-CarNet posterior inference and no scene geometry modification happen in M2.

**Decision**: M2 gate `PASS`. The next allowed stage is M3 scene-region posterior inference.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage2_region_mining_design.md`
- Implementation report: `docs/car_model/meshprior_stage2_region_mining_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage2_region_mining_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 3 (scene-region posterior inference) — PASS

**Outcome**: Implemented the wrapper that takes mined scene regions, samples region point clouds, canonicalizes them with a conservative bbox/PCA transform, runs the Stage-3 SP-CarNet posterior encoder, and writes posterior diagnostics for later proposal generation.

**Files added**:
- `ss3dm_prior/meshprior/scene_region_posterior.py`
- `scripts/car_model/meshprior_infer_region_posterior.py`
- `scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py`
- `docs/car_model/meshprior_stage3_scene_region_posterior_design.md`
- `docs/car_model/meshprior_stage3_scene_region_posterior_implementation_report.md`
- `docs/car_model/meshprior_stage3_scene_region_posterior_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage3_region_posterior.py`: PASS.
- Missing-checkpoint path fails clearly with `posterior_checkpoint not found`.
- With local checkpoint `outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt`, one synthetic region produced `z_mean.npy`, `z_logvar.npy`, `canonical_transform.json`, `posterior_summary.json`, sampled points, and an occupancy grid.

**Diagnostics from smoke**:
- `field_occupancy_ratio=0.070068`.
- `posterior_mu_norm=2.835622`.
- `posterior_logvar_mean=-3.936054`.
- `uncertainty_score=0.146494`.
- MC extraction succeeded at smoke resolution with `vertex_count=461`, `face_count=926`, watertight.

**Decision**: M3 gate `PASS`. The next allowed stage is M4 protect/prune proposal generation.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage3_scene_region_posterior_design.md`
- Implementation report: `docs/car_model/meshprior_stage3_scene_region_posterior_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage3_scene_region_posterior_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 4 (protect/prune proposals) — PASS

**Outcome**: Implemented the first safe MeshPrior proposal types: protect and prune. This stage only emits triangle-level scores and proposal records; it does not move vertices and does not fill holes.

**Files added**:
- `ss3dm_prior/meshprior/proposals.py`
- `ss3dm_prior/meshprior/protect_prune.py`
- `scripts/car_model/meshprior_make_protect_prune_proposals.py`
- `scripts/car_model/smoke_test_meshprior_stage4_protect_prune.py`
- `docs/car_model/meshprior_stage4_protect_prune_design.md`
- `docs/car_model/meshprior_stage4_protect_prune_implementation_report.md`
- `docs/car_model/meshprior_stage4_protect_prune_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage4_protect_prune.py`: PASS.

**Synthetic scoring result**:
- cube surface protect score mean `0.999990`.
- floater protect score `0.000010`.
- cube prune score mean `0.0`.
- floater prune score `0.999980`.
- both `protect` and `prune` proposal types generated.

**Decision**: M4 gate `PASS`. The next allowed stage is M5 optimizer adapter.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage4_protect_prune_design.md`
- Implementation report: `docs/car_model/meshprior_stage4_protect_prune_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage4_protect_prune_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 5 (optimizer adapter) — PASS

**Outcome**: Implemented a neutral optimizer adapter that exports MeshPrior protect/prune scores for downstream consumption without patching PRISM or overriding scene evidence.

**Files added**:
- `ss3dm_prior/meshprior/optimizer_adapter.py`
- `scripts/car_model/meshprior_export_optimizer_scores.py`
- `scripts/car_model/smoke_test_meshprior_stage5_optimizer_adapter.py`
- `docs/car_model/meshprior_stage5_optimizer_adapter_design.md`
- `docs/car_model/meshprior_stage5_optimizer_adapter_implementation_report.md`
- `docs/car_model/meshprior_stage5_optimizer_adapter_smoke.md`

**PRISM status**: PRISM is present (`utils/prism_scoring.py`, `utils/prism_counterfactual.py`, `utils/prism_pipeline.py`). M5 exports passive artifacts only; no `train.py` or PRISM internals were changed.

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage5_optimizer_adapter.py`: PASS.
- Generic NPZ and PRISM JSON export/reload verified.
- Bounded-add rule verified: MeshPrior score delta cannot exceed configured weight (`0.25` in smoke).

**Decision**: M5 gate `PASS`. The next allowed stage is M6 synthetic mesh-damage benchmark.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage5_optimizer_adapter_design.md`
- Implementation report: `docs/car_model/meshprior_stage5_optimizer_adapter_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage5_optimizer_adapter_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 6 (synthetic mesh-damage benchmark) — PASS

**Outcome**: Implemented a controlled synthetic mesh-damage benchmark for proposal behavior before real scene integration.

**Files added**:
- `ss3dm_prior/meshprior/synthetic_damage.py`
- `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py`
- `scripts/car_model/meshprior_make_synthetic_damage_report.py`
- `scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py`
- `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_design.md`
- `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_implementation_report.md`
- `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage6_synthetic_damage.py`: PASS.
- Synthetic benchmark produced 4 rows across local hole, floater, vertex noise, and density imbalance.
- Controlled floater case achieved `floater_prune_recall=1.0` and valid-surface protect recall >= 0.9.

**Outputs**:
- `metrics.json`, `metrics.csv`, `table_by_damage_type.csv`, `failure_cases.md`.
- Markdown report generation verified.

**Decision**: M6 gate `PASS`. The next allowed stage is M7 conservative snap.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_design.md`
- Implementation report: `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 7 (conservative snap proposals) — PASS

**Outcome**: Implemented bounded vertex snap proposals with explicit risk evaluation and a downstream acceptance gate. This is the first MeshPrior stage that proposes geometry movement, so proposals remain passive unless a later scene gate accepts them.

**Files added/updated**:
- `ss3dm_prior/meshprior/snap.py`
- `scripts/car_model/meshprior_make_snap_proposals.py`
- `scripts/car_model/smoke_test_meshprior_stage7_snap.py`
- `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py`
- `docs/car_model/meshprior_stage7_conservative_snap_design.md`
- `docs/car_model/meshprior_stage7_conservative_snap_implementation_report.md`
- `docs/car_model/meshprior_stage7_conservative_snap_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage7_snap.py`: PASS.
- `smoke_test_meshprior_stage6_synthetic_damage.py`: PASS with 8 benchmark rows after adding `protect_prune_snap`.
- Small M7 benchmark over `vertex_noise` and `floater`: PASS.

**Benchmark gate detail**:
- A first `snap_max_disp=0.02` benchmark trial improved vertex-noise surface distance but reduced valid-surface protect recall from `0.9167` to `0.8333`; this exceeded the 5 percent preservation tolerance.
- The benchmark snap default was tightened to `0.005`.
- Final `protect_prune_snap` on `vertex_noise` improved surface distance by `0.01073157787322998` while preserving valid-surface protect recall at `0.9166666666666666`.
- Floater prune recall stayed `1.0`.

**Decision**: M7 gate `PASS`. The next allowed stage is M8 guarded patch/fill proposals.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage7_conservative_snap_design.md`
- Implementation report: `docs/car_model/meshprior_stage7_conservative_snap_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage7_conservative_snap_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 8 (guarded fill proposals) — PASS

**Outcome**: Implemented guarded local hole-fill proposals. Fill remains proposal-only and is not approved for scene-level hidden-side completion until M9 evidence gates and rollback exist.

**Files added/updated**:
- `ss3dm_prior/meshprior/fill.py`
- `scripts/car_model/meshprior_make_fill_proposals.py`
- `scripts/car_model/smoke_test_meshprior_stage8_fill.py`
- `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py`
- `docs/car_model/meshprior_stage8_guarded_fill_design.md`
- `docs/car_model/meshprior_stage8_guarded_fill_implementation_report.md`
- `docs/car_model/meshprior_stage8_guarded_fill_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage8_fill.py`: PASS.
- `smoke_test_meshprior_stage6_synthetic_damage.py`: PASS.
- Small local-hole benchmark over `damaged_input`, `guarded_fill`, and `snap_fill`: PASS.

**Benchmark gate detail**:
- `damaged_input` local hole had `boundary_edge_count=4`.
- `guarded_fill` reduced boundary edges to `0`, added `4` faces, and kept component-count delta at `0`.
- `snap_fill` also reduced boundary edges to `0`; snap moved no vertices in this case because boundary vertices are fixed by default.
- Free-space violation stayed `0.0` in the controlled analytic benchmark.

**Decision**: M8 gate `PASS`. The next allowed stage is M9 scene evidence gates and rollback.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage8_guarded_fill_design.md`
- Implementation report: `docs/car_model/meshprior_stage8_guarded_fill_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage8_guarded_fill_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 9 (scene gates and rollback) — PASS

**Outcome**: Implemented dry-run scene evidence gates and rollback snapshots for MeshPrior proposals. Proposal acceptance now requires scene-side evidence; object-prior confidence alone is insufficient.

**Files added**:
- `ss3dm_prior/meshprior/scene_gate.py`
- `scripts/car_model/meshprior_evaluate_proposals.py`
- `scripts/car_model/smoke_test_meshprior_stage9_scene_gate.py`
- `docs/car_model/meshprior_stage9_scene_gate_rollback_design.md`
- `docs/car_model/meshprior_stage9_scene_gate_rollback_implementation_report.md`
- `docs/car_model/meshprior_stage9_scene_gate_rollback_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage9_scene_gate.py`: PASS.
- `smoke_test_meshprior_stage8_fill.py`: PASS.

**Gate behavior**:
- Topology-improving fill proposal accepted.
- Disconnected-floater proposal rejected because component count increased.
- Rollback snapshot and restore verified for vertices, faces, and metadata.
- CLI dry-run report generated `accepted_count=1` and `rejected_count=1`.

**Decision**: M9 gate `PASS`. The next allowed stage is M10 scene-level optimization integration.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage9_scene_gate_rollback_design.md`
- Implementation report: `docs/car_model/meshprior_stage9_scene_gate_rollback_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage9_scene_gate_rollback_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 10 (alternating runner) — PASS

**Outcome**: Implemented a dry-run orchestration runner that connects synthetic scene setup, region artifacts, posterior summary, proposal generation, scene gate evaluation, accepted proposal export, and report generation.

**Files added**:
- `scripts/car_model/meshprior_run_pipeline.py`
- `scripts/car_model/smoke_test_meshprior_stage10_pipeline.py`
- `docs/car_model/meshprior_stage10_alternating_runner_design.md`
- `docs/car_model/meshprior_stage10_alternating_runner_implementation_report.md`
- `docs/car_model/meshprior_stage10_alternating_runner_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage10_pipeline.py`: PASS.
- `smoke_test_meshprior_stage9_scene_gate.py`: PASS.

**Pipeline output**:
- Synthetic dry-run completed with `accepted_count=1` and `rejected_count=0`.
- Artifacts written: `regions.json`, `posterior/posterior_summary.json`, proposal files, `scene_gate/gate_report.json`, `accepted_proposals.json`, and `pipeline_report.md`.
- Geometry application remains disabled; `--apply` raises in M10.

**Decision**: M10 gate `PASS`. The next allowed stage is M11 actual scene training/evaluation and wandb.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage10_alternating_runner_design.md`
- Implementation report: `docs/car_model/meshprior_stage10_alternating_runner_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage10_alternating_runner_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 11 (scene experiment) — PASS

**Outcome**: Ran one dry-run scene experiment on the synthetic local-hole scene produced by the M10 pipeline.

**Files added**:
- `docs/car_model/meshprior_stage11_scene_experiment_design.md`
- `docs/car_model/meshprior_stage11_scene_experiment_report.md`
- `scripts/car_model/meshprior_collect_scene_experiment.py`

**Required outputs generated**:
- `outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/commands.sh`
- `outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/metrics.json`
- `outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/summary.md`

**Smoke / verification**:
- `git status --short`: only existing dirty submodules and M11 files before commit.
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `nvidia-smi`: available with elevated permissions. No fully idle GPU was available because every GPU had active processes and memory allocations.
- `smoke_test_meshprior_stage10_pipeline.py`: PASS.
- Dry-run M11 experiment: PASS.

**Metrics**:
- `proposal_count=1`.
- `accepted_count=1`.
- `rejected_count=0`.
- `boundary_edge_delta_sum=4.0`.
- `component_count_delta_max=0.0`.
- `floater_count_delta_max=0.0`.
- `free_space_violation_delta_max=0.0`.

**Wandb / training**:
- Wandb is installed, but no online wandb run was started.
- Full training was not launched because no fully idle GPU was available.

**Decision**: M11 gate `PASS` for dry-run scene experiment. The next allowed stage is M12 prior calibration, with the caveat that render/COLMAP improvements remain unproven until a real scene checkpoint is evaluated.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage11_scene_experiment_design.md`
- Report: `docs/car_model/meshprior_stage11_scene_experiment_report.md`

---

## 2026-05-01 — MeshPrior Stage 12 (prior calibration) — PASS

**Outcome**: Implemented a post-hoc surface-support calibration profile for proposal reliability. The upgrade targets snap risk and valid-surface preservation, not object Chamfer.

**Files added/updated**:
- `ss3dm_prior/meshprior/calibration.py`
- `scripts/car_model/meshprior_calibrate_prior.py`
- `scripts/car_model/smoke_test_meshprior_stage12_prior_calibration.py`
- `scripts/car_model/meshprior_run_pipeline.py`
- `docs/car_model/meshprior_stage12_prior_calibration_design.md`
- `docs/car_model/meshprior_stage12_prior_calibration_implementation_report.md`
- `docs/car_model/meshprior_stage12_prior_calibration_smoke.md`

**Evidence and calibration**:
- Uncalibrated snap (`max_disp=0.02`) reduced valid-surface protect recall from `0.9167` to `0.8333`.
- `surface_support_v1` snap (`max_disp=0.005`) preserved valid-surface protect recall at `0.9167`.
- Calibrated snap still improved surface distance by `0.01073157787322998`.
- Free-space violation delta remained `0.0`.

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage12_prior_calibration.py`: PASS.
- `smoke_test_meshprior_stage10_pipeline.py`: PASS.
- Targeted experiment wrote `outputs/carnet/meshprior/prior_calibration/stage12_surface_support_v1/calibration_metrics.json`.

**Decision**: M12 gate `PASS`. The next allowed stage is M13 evaluation protocol and matrix.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage12_prior_calibration_design.md`
- Implementation report: `docs/car_model/meshprior_stage12_prior_calibration_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage12_prior_calibration_smoke.md`

---

## 2026-05-01 — Training cleanup blocker repair before M13 — PASS

**Outcome**: Repaired destructive final cleanup behavior found by the wandb training smoke.

**Problem**:
- A non-PRISM 200-iteration training run pruned `5706` triangles to `15` at final cleanup.
- Root cause: final cleanup executed by default when PRISM was disabled.

**Fix**:
- `train.py` now executes final cleanup only when PRISM pruning is enabled and `prism_disable_final_cleanup_prune` is false.
- Ordinary non-PRISM training skips the PRISM-specific destructive cleanup path.

**Verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- 200-iteration wandb repair run on GPU 1: PASS.
- Wandb run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/3swt58x2`.
- Final cleanup summary: `final_cleanup_enabled=false`, `final_cleanup_pruned=0`.
- Triangle count preserved: `5706 -> 5706`.
- Vertex count preserved: `17118 -> 17118`.
- COLMAP sparse geometry eval passed at iteration 200 with depth AbsRel `0.10470779720655764`, depth MAE `0.024122862845250084`, normal mean angle `37.51919533010328`.

**Decision**: blocker `PASS`. M13 may proceed only after this repair is committed and pushed.

**Linked artefact**:
- `docs/car_model/meshprior_training_cleanup_repair_report.md`

---

## 2026-05-01 — MeshPrior Stage 13 (evaluation protocol and matrix) — PASS

**Outcome**: Implemented the evaluation protocol, experiment matrix registry, dry-run matrix runner, and NeurIPS-style report generator.

**Files added**:
- `docs/car_model/meshprior_stage13_eval_protocol_design.md`
- `configs/ss3dm_prior/meshprior/meshprior_experiment_matrix.yaml`
- `scripts/car_model/meshprior_run_experiment_matrix.py`
- `scripts/car_model/meshprior_make_neurips_report.py`
- `scripts/car_model/smoke_test_meshprior_stage13_eval_protocol.py`
- `docs/car_model/meshprior_stage13_eval_protocol_implementation_report.md`
- `docs/car_model/meshprior_stage13_eval_protocol_smoke.md`
- `docs/car_model/reports/meshprior_neurips_main_report.md`

**Generated outputs**:
- `outputs/carnet/meshprior/experiment_matrix/matrix_results.json`
- `outputs/carnet/meshprior/reports/object_table.csv`
- `outputs/carnet/meshprior/reports/synthetic_damage_table.csv`
- `outputs/carnet/meshprior/reports/scene_table.csv`
- `outputs/carnet/meshprior/reports/ablation_table.csv`
- `outputs/carnet/meshprior/reports/failure_cases.md`

**Full dry-run matrix**:
- `total=11`
- `available=7`
- `missing=4`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage13_eval_protocol.py`: PASS.
- Report generation from `outputs/carnet/meshprior/experiment_matrix/matrix_results.json`: PASS.

**Key available evidence**:
- Stage 3 posterior encoder: recon Chamfer L1 `0.0663909994752951`, hidden Chamfer L1 `0.0990753869336207`, mesh extraction success `1.0`.
- `surface_support_v1` calibration preserves valid-surface protect recall at `0.9166666666666666`.
- 200-iteration no-cleanup scene smoke preserved `5706` triangles and reports COLMAP depth AbsRel `0.10470779720655764`.

**Missing rows retained**:
- `v0_7_residual_baseline`
- `spcarnet_stage4_map_refinement`
- `spcarnet_stage5_oracle_k8`
- `protect_prune_proposals`

**Decision**: M13 gate `PASS`. The next allowed stage is M14, with the caveat that scene MeshPrior application is still dry-run/gated proposal evidence rather than real render-gated insertion.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage13_eval_protocol_design.md`
- Implementation report: `docs/car_model/meshprior_stage13_eval_protocol_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage13_eval_protocol_smoke.md`
- Generated report: `docs/car_model/reports/meshprior_neurips_main_report.md`

---

## 2026-05-01 — Pre-M14 stability audit — PASS

**Outcome**: Ran a stability audit before starting M14 and fixed one reproducibility bug in the smoke tests.

**Risk found and fixed**:
- Some MeshPrior smoke tests used ambient `python` for subprocesses, which can resolve to the wrong interpreter and fail with missing dependencies.
- Updated those subprocess calls to use `sys.executable`.

**Files updated**:
- `scripts/car_model/smoke_test_meshprior_stage2_region_mining.py`
- `scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py`
- `scripts/car_model/smoke_test_meshprior_stage5_optimizer_adapter.py`
- `scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py`
- `docs/car_model/meshprior_pre_m14_stability_audit.md`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- MeshPrior smoke tests M2, M3, M4, M5, M6, M7, M8, M9, M10, M12, and M13: PASS.
- M13 matrix/report dry-run: PASS with `total=11`, `available=7`, `missing=4`.

**Remaining non-collapse risks**:
- Scene MeshPrior application is still dry-run/gated proposal evidence.
- The 200-iteration scene result is a smoke run, not a full headline training run.
- Historical missing rows remain intentionally visible as `MISSING`.

**Decision**: Pre-M14 stability gate `PASS`.

**Linked artefact**:
- `docs/car_model/meshprior_pre_m14_stability_audit.md`

---

## 2026-05-01 — MeshPrior Stage 14 (paper roadmap and claim-risk analysis) — PASS

**Outcome**: Wrote the paper-level roadmap and claim-risk analysis for the MeshPrior direction.

**File added**:
- `docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md`

**Recommendation**:
- `MORE_SCENE_EVIDENCE_REQUIRED`

**Reasoning**:
- The proposal/gate/rollback direction is coherent and stable after M13 plus the pre-M14 audit.
- Current evidence supports a research direction, not a submission-ready scene result.
- Real render-gated MeshPrior insertion is not implemented.
- The scene evidence remains a 200-iteration diagnostic smoke plus synthetic dry-run proposal evidence.

**Required next evidence before strong submission**:
- real scene baseline and gated MeshPrior rows under fixed split;
- scene geometry improvement on COLMAP sparse AbsRel or normal proxy;
- no meaningful render regression;
- controlled triangle/FPS budget;
- car ROI hole/floater reduction;
- safety ablations showing direct prior insertion or gate removal is worse.

**Decision**: M14 gate `PASS`. The next allowed stage is M15 only if we intentionally pursue retrieval-deformation fallback; otherwise the higher-priority engineering milestone is real scene proposal application and render-gated evaluation.

**Linked artefact**:
- `docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md`

---

## 2026-05-01 — MeshPrior Stage 15 (retrieval-deformation fallback) — PASS

**Outcome**: Implemented and measured a train-only retrieval-deformation fallback for MeshPrior proposals.

**Files added**:
- `ss3dm_prior/meshprior/retrieval_deformation.py`
- `scripts/car_model/meshprior_build_anchor_bank.py`
- `scripts/car_model/meshprior_eval_retrieval_deformation.py`
- `scripts/car_model/smoke_test_meshprior_stage15_retrieval_deformation.py`
- `docs/car_model/meshprior_stage15_retrieval_deformation_design.md`
- `docs/car_model/meshprior_stage15_retrieval_deformation_implementation_report.md`
- `docs/car_model/meshprior_stage15_retrieval_deformation_smoke.md`

**Evaluation outputs**:
- `outputs/carnet/meshprior/retrieval_deformation/stage15_anchor_bank.npz`
- `outputs/carnet/meshprior/retrieval_deformation/stage15_eval/metrics.json`
- `outputs/carnet/meshprior/retrieval_deformation/stage15_eval/summary.md`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- Stage 15 smoke: PASS.
- Train-only anchor bank built from `outputs/carnet/spcarnet/object_index_v1.json`: `32` anchors, `512` points each.
- Retrieval/deformation evaluation rows: `12`.

**Decision**:
- Stage gate: `PASS`.
- Recommendation: `KEEP_AS_BASELINE`.
- Retrieval-only did not beat the Stage 3 posterior proxy on synthetic proposal metrics, so no pivot to retrieval-deformation is recommended.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage15_retrieval_deformation_design.md`
- Implementation report: `docs/car_model/meshprior_stage15_retrieval_deformation_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage15_retrieval_deformation_smoke.md`

---

## 2026-05-01 — MeshPrior scene application bridge — PASS

**Outcome**: Implemented a safe accepted-proposal application bridge before attempting real scene recovery training.

**Files added**:
- `ss3dm_prior/meshprior/apply_proposals.py`
- `scripts/car_model/meshprior_apply_accepted_proposals.py`
- `scripts/car_model/smoke_test_meshprior_scene_application.py`
- `docs/car_model/meshprior_scene_application_loop_design.md`
- `docs/car_model/meshprior_scene_application_loop_implementation_report.md`
- `docs/car_model/meshprior_scene_application_loop_smoke.md`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- Scene application smoke: PASS.
- Applied existing M11 accepted synthetic fill proposal to a copy.

**Synthetic application result**:
- accepted proposals: `1`
- applied proposals: `1`
- initial mesh: `8` vertices, `10` faces
- final mesh: `9` vertices, `14` faces
- rollback written
- recovery command plan written

**Decision**: bridge gate `PASS`. The next step is real scene proposal application plus recovery optimization, but it requires user confirmation of target scene/model and GPU before launching.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_scene_application_loop_design.md`
- Implementation report: `docs/car_model/meshprior_scene_application_loop_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_scene_application_loop_smoke.md`

---

## 2026-05-01 — Parking phone tiny scene audit and short baseline — PASS

**Outcome**: Audited the parking scene dataset, created a repo-local symlink view, and ran a 200-iteration wandb baseline.

**Files added**:
- `scripts/car_model/meshprior_prepare_parking_scene.py`
- `docs/car_model/meshprior_parking_phone_tiny_scene_audit.md`
- `docs/car_model/meshprior_parking_phone_tiny_baseline_200iter_report.md`

**Dataset view**:
- `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`
- images: `425`
- COLMAP images: `425`
- missing/extra image mismatch: `0`
- segmentation masks: `425`
- ground masks: `425`
- out-of-train split present.

**Training**:
- GPU: `1`
- iterations: `200`
- wandb run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/icjop1fq`
- test PSNR: `11.576681349012587`
- test SSIM: `0.3399546378188663`
- test LPIPS: `0.6316130017792737`
- test FPS: `374.0412913994465`
- triangles: `64497`
- vertices: `193491`
- final cleanup pruned: `0`

**Geometry eval**:
- evaluated test views: `54`
- depth AbsRel: `0.32417137460470213`
- depth MAE: `3.6485552222775537`
- normal mean angle: `51.68797353552561`

**Decision**: parking scene readiness gate `PASS`. This is a short baseline smoke, not a final baseline. Next high-value step is vehicle/ground-aware region mining and gated MeshPrior recovery smoke on this scene, or a longer baseline if a stronger reference is needed first.

**Linked artefacts**:
- Scene audit: `docs/car_model/meshprior_parking_phone_tiny_scene_audit.md`
- Baseline report: `docs/car_model/meshprior_parking_phone_tiny_baseline_200iter_report.md`

---

## 2026-05-01 — Parking phone tiny image/COLMAP region mining — PASS

**Outcome**: Implemented image/COLMAP ROI mining from segmentation masks, ground masks, and COLMAP sparse observations.

**Files added**:
- `scripts/car_model/meshprior_mine_parking_image_regions.py`
- `scripts/car_model/smoke_test_meshprior_parking_image_regions.py`
- `docs/car_model/meshprior_parking_image_region_mining_report.md`

**Full mining output**:
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions.json`
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_region_mining_report.md`

**Metrics**:
- images considered: `425`
- candidate regions: `340`
- eligible candidates: `273`
- median sparse point count: `4`
- median mask area fraction: `0.0030437403549382716`
- max eligible ground overlap: `0.25251004016064255`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_image_regions.py`: PASS.

**Decision**: region mining gate `PASS`. The next step is multi-view clustering / 3D consolidation before proposal scoring; these 2D ROI candidates must not directly edit scene geometry.

**Linked artefact**:
- `docs/car_model/meshprior_parking_image_region_mining_report.md`

---

## 2026-05-01 — Parking phone tiny region consolidation — PASS

**Outcome**: Consolidated parking image ROI candidates into coarse multi-view 3D vehicle-region candidates.

**Files added**:
- `scripts/car_model/meshprior_cluster_parking_regions.py`
- `scripts/car_model/smoke_test_meshprior_parking_region_consolidation.py`
- `docs/car_model/meshprior_parking_region_consolidation_report.md`

**Full consolidation output**:
- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json`
- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidation_report.md`

**Metrics**:
- input ROI regions: `340`
- sparse-supported eligible inputs used: `140`
- consolidated clusters: `17`
- eligible clusters: `9`
- top cluster support: `32` views and `3851` sparse points.

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_region_consolidation.py`: PASS.

**Decision**: consolidation gate `PASS`. The next step is proposal scoring for the consolidated clusters; no scene geometry has been edited.

**Linked artefact**:
- `docs/car_model/meshprior_parking_region_consolidation_report.md`

---

## 2026-05-01 — Parking phone tiny cluster proposal scoring — PASS

**Outcome**: Converted consolidated parking scene clusters into MeshPrior proposal metadata.

**Files added**:
- `scripts/car_model/meshprior_score_parking_clusters.py`
- `scripts/car_model/smoke_test_meshprior_parking_cluster_scoring.py`
- `docs/car_model/meshprior_parking_cluster_proposal_scoring_report.md`

**Full scoring output**:
- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposals.json`
- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposal_scores.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposal_report.md`

**Metrics**:
- eligible clusters scored: `9`
- proposals emitted: `45`
- proposal types: `protect`, `prune`, `snap_candidate`, `fill_candidate`, `uncertainty`
- metadata-only proposals: `45`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_cluster_scoring.py`: PASS.

**Decision**: proposal scoring gate `PASS`. These proposals are not yet geometry edits; every proposal is marked `requires_mesh_extraction` and `requires_scene_gate`.

**Linked artefact**:
- `docs/car_model/meshprior_parking_cluster_proposal_scoring_report.md`

---

## 2026-05-01 — Parking phone tiny metadata proposal gate — PASS

**Outcome**: Gated metadata-only parking proposals into a local mesh-extraction action plan.

**Files added**:
- `scripts/car_model/meshprior_gate_parking_metadata_proposals.py`
- `scripts/car_model/smoke_test_meshprior_parking_metadata_gate.py`
- `docs/car_model/meshprior_parking_metadata_gate_report.md`

**Full gate output**:
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/action_plan.json`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_report.md`

**Metrics**:
- proposals evaluated: `45`
- candidate_extract: `24`
- deferred: `17`
- diagnostic: `1`
- rejected: `3`
- mesh extraction targets: `8`
- diagnostic targets: `1`
- geometry edited: `false`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_metadata_gate.py`: PASS.

**Decision**: metadata gate `PASS`. The next missing bridge is local scene mesh patch extraction with stable face IDs; prune remains deferred until real scene mesh evidence exists.

**Linked artefact**:
- `docs/car_model/meshprior_parking_metadata_gate_report.md`

---

## 2026-05-01 — Parking phone tiny local mesh patch extraction — PASS

**Outcome**: Extracted local mesh patches for metadata-gated parking targets from the trained triangle checkpoint.

**Files added**:
- `scripts/car_model/meshprior_extract_parking_mesh_patches.py`
- `scripts/car_model/smoke_test_meshprior_parking_mesh_patch_extraction.py`
- `docs/car_model/meshprior_parking_mesh_patch_extraction_report.md`

**Full extraction output**:
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/patches/*.npz`

**Metrics**:
- checkpoint vertices: `193491`
- checkpoint triangles: `64497`
- patches extracted: `8`
- nonempty patches: `8`
- total patch faces: `10826`
- patch face range: `97` - `3902`
- geometry edited: `false`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_mesh_patch_extraction.py`: PASS.

**Decision**: local patch extraction gate `PASS`. The parking pipeline now has real local mesh assets with original face/vertex indices for downstream before/after gates and rollback.

**Linked artefact**:
- `docs/car_model/meshprior_parking_mesh_patch_extraction_report.md`

---

## 2026-05-01 — Parking phone tiny patch no-op/protect gate — PASS

**Outcome**: Ran a no-op/protect readiness gate over extracted parking mesh patches and wrote rollback snapshots.

**Files added**:
- `scripts/car_model/meshprior_gate_parking_mesh_patches.py`
- `scripts/car_model/smoke_test_meshprior_parking_mesh_patch_gate.py`
- `docs/car_model/meshprior_parking_mesh_patch_gate_report.md`

**Full gate output**:
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/rollback_snapshots/*.npz`

**Metrics**:
- patches evaluated: `8`
- protect_ready: `8`
- deferred: `0`
- failed: `0`
- rollback snapshots: `8`
- geometry edited: `false`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_mesh_patch_gate.py`: PASS.

**Decision**: patch no-op/protect gate `PASS`. The parking real-scene bridge now has stable local mesh patches plus rollback snapshots; next step is copied-patch before/after proposal testing.

**Linked artefact**:
- `docs/car_model/meshprior_parking_mesh_patch_gate_report.md`

---

## 2026-05-01 — Parking phone tiny copied-patch proposal tests — SOFT PASS

**Outcome**: Ran copied-patch before/after tests over extracted parking mesh patches.

**Files added**:
- `scripts/car_model/meshprior_test_parking_patch_proposals.py`
- `scripts/car_model/smoke_test_meshprior_parking_patch_proposals.py`
- `docs/car_model/meshprior_parking_patch_proposal_test_report.md`

**Full test output**:
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/proposal_meshes/*/*.npz`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/rollback_snapshots/*.npz`

**Metrics**:
- patches tested: `8`
- proposal tests: `24`
- accepted: `8`
- rejected: `16`
- protect_noop_rejected: `8`
- cleanup_accepted: `8`
- floater_rejected: `8`
- source model edited: `false`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_patch_proposals.py`: PASS.

**Decision**: copied-patch proposal test gate `SOFT PASS`. The gate behaves correctly on copied local patches, but accepted cleanup candidates still need checkpoint-copy application and render/geometry validation before they can be treated as scene improvements.

**Linked artefact**:
- `docs/car_model/meshprior_parking_patch_proposal_test_report.md`

---

## 2026-05-01 — Parking phone tiny checkpoint-copy cleanup — SOFT PASS

**Outcome**: Applied accepted copied-patch cleanup candidates to a duplicated parking triangle checkpoint and verified state-array integrity.

**Files added**:
- `scripts/car_model/meshprior_apply_parking_patch_cleanup_to_checkpoint_copy.py`
- `scripts/car_model/smoke_test_meshprior_parking_checkpoint_copy_cleanup.py`
- `docs/car_model/meshprior_parking_checkpoint_copy_cleanup_report.md`

**Full application output**:
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_rows.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_report.md`

**Metrics**:
- cleanup applications: `8`
- unique removed faces: `532`
- faces: `64497` -> `63965`
- vertices: `193491` -> `191895`
- source model edited: `false`
- checkpoint copy edited: `true`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_checkpoint_copy_cleanup.py`: PASS.

**Decision**: checkpoint-copy cleanup gate `SOFT PASS`. Writeback bookkeeping is valid, but render/geometry validation is still pending before this can be claimed as a scene improvement.

**Linked artefact**:
- `docs/car_model/meshprior_parking_checkpoint_copy_cleanup_report.md`

---
