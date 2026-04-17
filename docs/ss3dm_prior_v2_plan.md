# SS3DM Prior V2 Plan

## Goal

Upgrade the current v1 local geometry prior into a v2 track with:

- visibility-aware corruption and conditioning
- prototype-memory retrieval support
- implicit local surface field support

Step 1 is intentionally limited to protocol hygiene and baseline cleanup. No visibility query path or new model blocks are introduced here.

## Steps

| Step | Scope | Status |
| --- | --- | --- |
| Step 1 | protocol audit, strict baseline config, retrieval metric repair, protocol warnings in eval/report | done |
| Step 2 | patch cache v2: visibility/free-space queries, intrinsic difficulty target, checker tooling | done |
| Step 3 | hybrid v2 model, prototype memory, occupancy head, intrinsic difficulty head, v2 losses/metrics | done |
| Step 4 | trainer/train loop/wandb logging for hybrid v2 | done |
| Step 5+ | pending user instruction | pending |

## Step 1 Decisions

- Keep v1 training and eval paths intact, but surface protocol risk explicitly instead of silently treating debug-friendly runs as formal baselines.
- Preserve the old retrieval metric as a renamed legacy metric: `retrieval_top1_self_aligned` and `retrieval_top5_self_aligned`.
- Add filtered neighbor metrics for non-self retrieval so future prototype-memory work can compare corrupted-query neighborhoods against clean-anchor neighborhoods without requiring a new model in Step 1.
- Add a new strict control baseline config `train_v8_strict.yaml` that explicitly disables:
  - `debug_use_all_patches_for_train_val`
  - `allow_debug_split_override`
  - `allow_split_fallback`

## Current Step 1 Outcome

- `train_full_v5` protocol audit is not valid under the strict protocol because saved metadata shows:
  - `debug_use_all_patches_for_train_val=true`
  - `allow_split_fallback=true`
  - `allow_debug_split_override` could not be verified from saved metadata
- Eval outputs now export protocol audit data into both `metrics_summary.json` and `report.md`.
- The strict control baseline for later v2 ablations is now `configs/ss3dm_prior/train_v8_strict.yaml` with `28` epochs and all debug/fallback switches explicitly off.

## Step 2 Decisions

- Keep the v1 teacher patch builder and v1 cache checker intact. Step 2 adds parallel v2 tooling instead of mutating the legacy CLI path.
- Extend the patch cache format with versioned, optional fields so old `.npz` patches can still be read through `load_patch_npz`.
- Generate free-space supervision from real LiDAR rays/ranges rather than from the already fused observed point cloud, because the fused cache no longer contains segment occupancy information.
- Treat `unknown_query_points` as ignore-region supervision only: they are packed into `query_points_all` with `query_ignore_mask=true`.
- Add an intrinsic difficulty target based on a weighted combination of observed-clean mismatch, visibility coverage, geometry irregularity, support deficit, and free-space contradiction.

## Current Step 2 Outcome

- Added `patch_cache_format_version: 2` support in both patch files and patch index records.
- Added cached query/supervision fields:
  - `surface_query_points`
  - `surface_query_labels`
  - `free_query_points`
  - `free_query_labels`
  - `unknown_query_points`
  - `query_points_all`
  - `query_labels_all`
  - `query_ignore_mask`
  - `camera_support_count`
  - `lidar_support_count`
  - `visible_surface_fraction`
  - `free_space_fraction`
  - `unknown_fraction`
- Added `intrinsic_patch_difficulty_target` and `difficulty_components_json`.
- Added v2 build/check CLIs:
  - `ss3dm_prior.tools.build_teacher_patch_cache_v2`
  - `ss3dm_prior.tools.check_teacher_patch_cache_v2`
- Real debug build completed at:
  - `outputs/ss3dm_prior/teacher_patch_cache_v2_debug`
- Real checker output confirmed at:
  - `outputs/ss3dm_prior/teacher_patch_cache_v2_debug/visualizations_v2/Town01__1000_streetsurf__tile_000000.png`

## Step 3 Decisions

- Keep `ss3dm_prior.models.patch_denoiser.LocalPatchDenoiser` as the stable trainer import path, but turn it into a dispatcher controlled by `model_type`.
- Preserve `legacy_v1` behavior by routing old configs to the original denoiser implementation without changing trainer code.
- Add a new `hybrid_v2` path with:
  - corrupted / observed / clean encoders
  - fused latent projection
  - vector-quantized prototype memory
  - point reconstruction head
  - implicit occupancy head over `query_points_all`
  - intrinsic difficulty prediction head
- Make loss/metric code gracefully degrade when v2 batch fields are absent, so old patch caches and old configs still work.

## Current Step 3 Outcome

- Added:
  - `ss3dm_prior/models/vector_quantizer.py`
  - `ss3dm_prior/models/hybrid_patch_prior_v2.py`
- Extended stable model entrypoint:
  - `model_type: legacy_v1`
  - `model_type: hybrid_v2`
- Added v2 loss support for:
  - `corruption_score_loss`
  - `intrinsic_difficulty_loss`
  - `occupancy_bce_loss`
  - `free_space_violation_loss`
  - `vq_commitment_loss`
  - `prototype_diversity_loss`
- Added v2 metrics:
  - `occupancy_iou_visible`
  - `free_space_violation_rate`
  - `intrinsic_difficulty_mae`
  - `intrinsic_difficulty_spearman`
  - `prototype_usage_entropy`
- Added `configs/ss3dm_prior/model_v8_hybrid.yaml`
- Real v2 patch batch forward succeeded on:
  - `outputs/ss3dm_prior/teacher_patch_cache_v2_debug/patch_index.jsonl`

## Step 4 Decisions

- Keep trainer model-agnostic by checking `model_type` and available batch/model fields, rather than branching the whole loop into separate v1/v2 trainers.
- Use config-driven curriculum by muting subsets of loss weights before their stage start epochs, instead of hardcoding separate optimizer phases.
- Implement hard-example emphasis through weighted sampling from `intrinsic_patch_difficulty_target`, which keeps easy patches in the mix while still biasing toward harder geometry.
- Preserve existing qualitative images, but add v2-native static PNGs that expose visibility, free-space, prototype selection, and intrinsic difficulty.

## Current Step 4 Outcome

- Trainer now supports both:
  - `legacy_v1`
  - `hybrid_v2`
- Added curriculum control through `train_config.curriculum`.
- Added hard-example weighted sampling through `train_config.hard_example_sampling`.
- Added new checkpoint targets:
  - `best_composite.pt`
  - `best_visibility.pt`
- Added new validation metrics/logging for:
  - intrinsic difficulty
  - occupancy / free-space
  - prototype usage entropy
  - non-self retrieval
- Added new qualitative outputs:
  - `visibility_panel`
  - `hybrid_reconstruction_panel`
  - `sequence_visibility_map`
  - `prototype_usage_gallery`
- Real 1-epoch debug hybrid training succeeded at:
  - `outputs/ss3dm_prior/train_v8_hybrid_debug_step4`
- Real debug run produced 8 PNGs including:
  - `Town01__1000_streetsurf__tile_000000_visibility_panel.png`
  - `Town01__1000_streetsurf__tile_000000_hybrid_reconstruction_panel.png`
  - `Town01__1000_streetsurf_hardest_sequence_visibility_map.png`
  - `prototype_usage_gallery.png`
