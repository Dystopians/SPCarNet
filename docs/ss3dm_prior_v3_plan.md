# SS3DM Prior V3 Plan

## Goal

Upgrade the standalone `ss3dm_prior/` track into a publication-oriented v3 line with:

- a strict-valid publication-grade baseline
- a stronger deterministic local geometry prior
- an optional exploratory latent diffusion / flow matching branch

This track must remain isolated from mesh splatting main training. No root `train.py` integration is planned in v3.

## Steps

| Step | Scope | Status |
| --- | --- | --- |
| Step 1 | protocol hygiene, strict control baseline, reporting cleanup | done |
| Step 2 | v3 patch semantics, multi-scale cache, corruption curriculum, visible/hidden eval fields | done |
| Step 3 | v9_wide capacity baseline on stable hybrid path | done |
| Step 4 | v10 cross-attention hybrid on stable dispatch path | done |
| Step 5 | trainer stabilization, paper metrics, eval/report upgrade, ablation suite | done |
| Step 6 | optional latent flow exploratory branch on top of v10 backbone | done |

## Step 1 Decisions

- Treat `wandb` CLI overrides as opt-in only: config stays authoritative unless `--wandb_mode` is explicitly provided.
- Separate synthetic corruption difficulty from intrinsic geometry difficulty throughout loss weighting, summary JSON, and markdown report output.
- Make protocol audit fields first-class evaluation outputs so strict/debug/fallback status is visible in both machine-readable and human-readable artifacts.
- Add a new formal control config `configs/ss3dm_prior/train_v9_strict_control.yaml` as the v3 baseline launch point.

## Current Step 1 Outcome

- `train.py` now preserves config `wandb_enable` unless `--wandb_mode` is explicitly passed.
- Loss hard-example reweighting now supports `corruption`, `intrinsic`, and `blend` sources via config, defaulting to `blend`.
- Eval/report outputs now separate:
  - `mean_intrinsic_difficulty`
  - `mean_corruption_score_target`
- Protocol audit output now surfaces:
  - `protocol_valid`
  - `protocol_warnings`
  - `train_towns`
  - `val_towns`
  - `test_towns`
  - `strict_protocol_enabled`
  - debug/fallback flags
- Added v3 strict control baseline config and launch script.

## Step 2 Decisions

- Keep v3 cache generation on a new parallel path instead of mutating the existing v2 builder/checker entrypoints.
- Use deterministic multi-scale patch ids based on `sequence_id`, `tile_id`, `scale_id`, and radius token so repeated builds are stable.
- Keep visible/hidden supervision in cache space rather than trying to infer it online in the trainer.
- Add corruption curriculum only at dataset level for now; validation and eval remain fixed-corruption and do not depend on epoch.
- Extend eval/report outputs with visible and hidden geometry metrics without changing the model architecture.

## Current Step 2 Outcome

- Added patch cache v3 support with:
  - `scale_id`
  - `patch_radius_m`
  - `visible_clean_points`
  - `visible_clean_normals`
  - `hidden_clean_points`
  - `hidden_clean_normals`
  - `surface_support_mask`
  - `free_space_query_hard_negatives`
  - `patch_cache_format_version: 3`
- Added dataset-visible fields:
  - `visible_clean_points`
  - `hidden_clean_points`
  - `visible_support_fraction`
  - `hidden_surface_fraction`
  - `free_space_hard_negative_count`
- Added corruption schedule controls:
  - `severity_schedule.type`
  - `severity_schedule.start_scale`
  - `severity_schedule.end_scale`
  - `severity_schedule.warmup_epochs`
  - dataset `set_epoch(epoch)`
- Eval/report now expose:
  - `visible_recon_chamfer_l1`
  - `hidden_completion_chamfer_l1`
- Added v3 build/check CLIs:
  - `ss3dm_prior.tools.build_teacher_patch_cache_v3`
  - `ss3dm_prior.tools.check_teacher_patch_cache_v3`
- Real debug v3 build succeeded at:
  - `outputs/ss3dm_prior/teacher_patch_cache_v3_debug`
- Real checker PNGs were produced, including:
  - `Town01__1000_streetsurf__tile_000000__scale_01__r4p00m__semantics.png`
  - `Town01__1000_streetsurf__tile_000000__scale_02__r6p00m__semantics.png`
  - `Town01__1000_streetsurf__tile_000000__multiscale.png`

## Step 3 Decisions

- Keep the stable public model entrypoint as `LocalPatchDenoiser` and extend the existing dispatch with a new `model_type: hybrid_v2_wide`.
- Reuse the current hybrid forward output protocol so trainer/loss/metric code stays low-coupling and does not branch per model variant.
- Avoid architectural changes like attention or diffusion; only widen encoders, fusion MLPs, decoders, retrieval heads, and occupancy heads.
- Make wider MLP behavior configurable with optional layer norm, dropout, and residual MLP blocks, while keeping `legacy_v1` and `hybrid_v2` backward compatible.

## Current Step 3 Outcome

- Added `model_type: hybrid_v2_wide` to the stable dispatch entrypoint.
- Added configurable wider MLP support to the hybrid path through:
  - encoder hidden dims
  - fusion hidden dims
  - decoder hidden dims
  - per-head hidden dims
  - occupancy hidden dims
  - optional layer norm
  - optional dropout
  - optional residual MLP blocks
- Added wide baseline config and launch script:
  - `configs/ss3dm_prior/model_v9_wide.yaml`
  - `configs/ss3dm_prior/train_v9_wide.yaml`
  - `scripts/ss3dm_prior/train_v9_wide.sh`
- Added forward coverage for:
  - random-tensor `hybrid_v2_wide` forward
  - real v3 patch batch `hybrid_v2_wide` forward

## Step 4 Decisions

- Keep the stable public model entrypoint as `LocalPatchDenoiser` and extend dispatch again instead of introducing a separate trainer or eval path.
- Add low-coupling latent cross-attention blocks for local patch conditioning, rather than switching to full point-to-point transformer attention.
- Use latent queries to aggregate corrupted, observed, visible, hidden, and query token sets, so the model can reason over conditional geometry without quadratic all-pairs point attention.
- Preserve the Step 3 output schema and loss contract so existing trainer, eval, and reporting code remain compatible.

## Current Step 4 Outcome

- Added a new stable dispatch type:
  - `model_type: v10_cross_attention_hybrid`
- Added low-coupling attention modules:
  - `ss3dm_prior/models/attention_blocks.py`
  - `ss3dm_prior/models/cross_attention_patch_prior_v10.py`
- New architecture uses:
  - `num_latent_queries: 48`
  - `num_cross_attention_layers: 4`
  - `num_latent_self_attention_layers: 2`
  - `attention_heads: 8`
  - `ffn_dim: 1024`
  - `dropout: 0.1`
- Cross-attention context now supports:
  - corrupted patch tokens
  - observed patch tokens
  - visible clean token sets
  - hidden clean token sets
  - occupancy query token sets
  - prototype-conditioned latent refinement after VQ selection
- Trainer and eval remain API-compatible while now passing visible/hidden tensors through when available.
- Added new baseline assets:
  - `configs/ss3dm_prior/model_v10_crossattn.yaml`
  - `configs/ss3dm_prior/train_v10_crossattn.yaml`
  - `scripts/ss3dm_prior/train_v10_crossattn.sh`
- Added forward coverage for:
  - random-tensor `v10_cross_attention_hybrid` forward
  - real v3 patch batch `v10_cross_attention_hybrid` forward
- Parameter count increased from:
  - `hybrid_v2_wide`: `40052874`
  - `v10_cross_attention_hybrid`: `52579338`

## Step 5 Decisions

- Keep the existing training/eval entrypoints and extend them with config-driven stabilization features rather than creating a parallel trainer stack.
- Make EMA, grad accumulation, curriculum staging, corruption scheduling, and hard-example weighting coexist cleanly so both `v9_wide` and `v10_cross_attention_hybrid` can use the same trainer contract.
- Add a dedicated `best_paper.pt` checkpoint route so paper-facing model selection can be separated from reconstruction-only or visibility-only checkpoints.
- Treat Step 2 visible/hidden semantics as first-class evaluation targets by adding visible/hidden reconstruction metrics, free-space false-positive metrics, and intrinsic calibration metrics end-to-end through trainer, eval, and report artifacts.
- Replace the older ablation runner with a Step 5 v3 suite centered on `v8_strict_control`, `v9_wide`, `v10_crossattn`, and multiscale/curriculum ablations.

## Current Step 5 Outcome

- Trainer now supports:
  - `grad_accum_steps`
  - `ema.enable`
  - `ema.decay`
  - `ema.use_for_eval`
  - staged `recon_warmup_epochs`
  - coexisting hard-example sampler + hard-example loss reweighting
  - `best_paper.pt` checkpoint selection
- Eval/report now expose additional paper-facing metrics:
  - `visible_recon_normal_cosine`
  - `hidden_completion_gain`
  - `intrinsic_difficulty_calibration_mae`
  - `free_space_fp_rate`
  - plus previously added visible/hidden Chamfer metrics
- Added new qualitative artifact categories:
  - `visible_vs_hidden_panel`
  - `free_space_error_panel`
  - `difficulty_calibration_panel`
  - `prototype_usage_gallery`
  - hardest / best_gain / worst_gain `sequence_visibility_map`
- Added Step 5 ablation assets:
  - `ss3dm_prior/tools/run_ablation_suite.py`
  - `ss3dm_prior/tools/aggregate_ablation_results.py`
  - `scripts/ss3dm_prior/run_v3_ablation_suite.sh`
- Added Step 5 smoke coverage assets:
  - `tests/ss3dm_prior/test_trainer_v3_smoke.py`
  - `tests/ss3dm_prior/test_eval_v3_smoke.py`
  - `tests/ss3dm_prior/smoke_v3_utils.py`
- `configs/ss3dm_prior/train_v10_crossattn.yaml` now carries trainer-side Step 5 fields for EMA, grad accumulation, recon warmup, and paper checkpoint weights.

## Step 6 Decisions

- Keep the deterministic `v10_cross_attention_hybrid` path intact and add a separate `v11_latent_flow_hybrid` dispatch branch rather than mutating the baseline into a stochastic-only model.
- Use latent flow matching instead of raw point diffusion so generation stays in the existing conditional latent / implicit-field space and can reuse the current decoder, prototype/VQ path, and occupancy head.
- Generate only hidden-region completion residuals (with fallback to clean latent residuals when hidden supervision is absent), not whole-patch point clouds from noise.
- Push candidate sampling into eval-time only, with K-way reranking based on observed consistency, visible consistency, free-space penalty, and prototype consistency so paper reporting can compare deterministic vs stochastic behavior directly.

## Current Step 6 Outcome

- Added a new stable dispatch type:
  - `model_type: v11_latent_flow_hybrid`
- Added a new exploratory latent generator:
  - `ss3dm_prior/models/latent_flow_patch_prior_v11.py`
- `v11` reuses the `v10` conditional backbone and adds:
  - latent flow matching loss
  - stochastic latent candidate sampling
  - hidden-residual completion target
  - reuse of residual decoder and occupancy head for candidate reranking
- Eval/report now expose exploratory stochastic metrics:
  - `best_of_k_hidden_completion`
  - `mean_of_k_hidden_completion`
  - `sample_diversity`
  - `free_space_safe_best_of_k`
  - deterministic vs stochastic comparison tables for `K=1/4/8`
- Added candidate qualitative artifacts:
  - `*_stochastic_candidates.png`
- Added Step 6 assets:
  - `configs/ss3dm_prior/model_v11_latent_flow.yaml`
  - `configs/ss3dm_prior/train_v11_latent_flow.yaml`
  - `scripts/ss3dm_prior/train_v11_latent_flow.sh`
  - `tests/ss3dm_prior/test_model_v11_latent_flow_forward.py`
