# SS3DM Prior V2 Update Log

## Step 1 - Protocol Audit, Strict Baseline, Retrieval Metric Repair

### Modified Files

- `ss3dm_prior/tools/audit_run_protocol.py`
- `ss3dm_prior/metrics.py`
- `ss3dm_prior/eval.py`
- `ss3dm_prior/reporting.py`
- `ss3dm_prior/engine/trainer.py`
- `configs/ss3dm_prior/train_v8_strict.yaml`
- `scripts/ss3dm_prior/train_v8_strict.sh`
- `tests/ss3dm_prior/test_protocol_audit.py`
- `tests/ss3dm_prior/test_eval_smoke.py`
- `docs/ss3dm_prior_v2_plan.md`
- `docs/ss3dm_prior_v2_update_log.md`
- `docs/ss3dm_prior_v2_experiments.md`

### Design Rationale

- Add a reusable protocol audit helper first so eval/report can consume one canonical definition of protocol validity instead of duplicating flag checks.
- Keep legacy retrieval behavior visible, but rename it to `self_aligned` so the metric no longer looks like a real non-self retrieval benchmark.
- Introduce non-self and cross-sequence retrieval metrics as filtered-neighbor agreement metrics against the clean anchor neighborhood. This is low-coupling, works with the current saved embeddings, and leaves room for later prototype-memory extensions.
- Add a strict v8 baseline config so later v2 ablations have an explicit control run with all debug/fallback split escapes disabled.

### Actual Commands

- `python -m ss3dm_prior.tools.audit_run_protocol --help`
- `python -m ss3dm_prior.eval --help`
- `python -m ss3dm_prior.tools.audit_run_protocol "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_full_v5"`
- `pytest tests/ss3dm_prior/test_protocol_audit.py -q -k retrieval`
- `pytest tests/ss3dm_prior/test_protocol_audit.py -q`
- `pytest tests/ss3dm_prior/test_eval_smoke.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/metrics.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/eval.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/reporting.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/audit_run_protocol.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_protocol_audit.py"`

### Result

- Added `audit_run_protocol` CLI and helper API that accepts either a run directory or checkpoint and emits:
  - `protocol_valid`
  - `protocol_warnings`
  - `protocol_summary`
- `metrics_summary.json` now includes protocol audit fields.
- `report.md` now starts with a `Protocol Audit` section.
- Eval now supports `--fail_on_protocol_warning`.
- Legacy retrieval metrics are now exported as:
  - `retrieval_top1_self_aligned`
  - `retrieval_top5_self_aligned`
- Added new filtered metrics:
  - `retrieval_top1_nonself`
  - `retrieval_top5_nonself`
  - `retrieval_top1_cross_sequence`
- Added strict baseline config and launch script:
  - `configs/ss3dm_prior/train_v8_strict.yaml`
  - `scripts/ss3dm_prior/train_v8_strict.sh`
- Audit of `outputs/ss3dm_prior/train_full_v5` reported protocol invalid with these warnings:
  - debug leakage via `debug_use_all_patches_for_train_val=true`
  - `allow_split_fallback=true`
  - `allow_debug_split_override` could not be verified from saved metadata

### Risk / TODO

- Current saved v5 metadata does not always preserve `allow_debug_split_override`, so old runs can remain only partially auditable.
- `retrieval_top1_nonself` and `retrieval_top5_nonself` are now meaningful filtered-neighbor agreement metrics, but they are still embedding-space protocol metrics rather than a geometry-grounded retrieval benchmark.
- Tiny smoke datasets can make filtered non-self retrieval undefined because too few valid candidates remain; this currently degrades to warnings plus `NaN`, which is acceptable for Step 1 but should be documented in later ablations.

## Step 2 - Patch Cache V2 With Visibility, Free-Space, and Intrinsic Difficulty

### Modified Files

- `ss3dm_prior/data/patch_types.py`
- `ss3dm_prior/data/visibility_queries.py`
- `ss3dm_prior/data/teacher_patch_builder.py`
- `ss3dm_prior/tools/build_teacher_patch_cache_v2.py`
- `ss3dm_prior/tools/check_teacher_patch_cache_v2.py`
- `configs/ss3dm_prior/teacher_patch_v2.yaml`
- `tests/ss3dm_prior/test_visibility_queries.py`
- `docs/ss3dm_prior_v2_plan.md`
- `docs/ss3dm_prior_v2_update_log.md`
- `docs/ss3dm_prior_v2_experiments.md`

### Design Rationale

- Add a versioned v2 cache path instead of replacing the legacy v1 builder so existing training/eval code keeps working unchanged.
- Store visibility/free-space supervision directly in the patch cache because later v2 models should not have to regenerate expensive geometric query labels online.
- Use LiDAR ray segments for free-space supervision, because fused observed points alone do not preserve known-empty space.
- Keep `unknown_query_points` explicitly cached so later implicit-field training can ignore ambiguous regions without confusing them with negative occupancy.
- Make `intrinsic_patch_difficulty_target` coexist with the online corruption score: the former reflects geometry/support difficulty, while the latter still reflects denoising corruption difficulty.

### Actual Commands

- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v2 --help`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v2 --help`
- `pytest tests/ss3dm_prior/test_visibility_queries.py -q`
- `pytest tests/ss3dm_prior/test_teacher_patch_builder.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/data/patch_types.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/data/visibility_queries.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/data/teacher_patch_builder.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/build_teacher_patch_cache_v2.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/check_teacher_patch_cache_v2.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_visibility_queries.py"`
- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v2 --manifest "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" --split_config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/default_town_split.yaml" --config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/teacher_patch_v2.yaml" --observed_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache" --town_mesh_cache_root "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache" --out_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v2_debug" --subsets train --debug_max_sequences 1 --debug_max_tiles_per_sequence 1 --num_workers 1 --seed 0`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v2 --patch_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v2_debug" --num_visualizations 1 --seed 0`

### Result

- Added v2 query generation module `ss3dm_prior.data.visibility_queries`.
- Added v2 patch builder entrypoints in `teacher_patch_builder.py`:
  - `build_patch_from_tile_v2`
  - `build_teacher_patches_for_sequence_v2`
- Added backward-compatible optional fields to patch `.npz` loading/saving and patch index records.
- Added cached supervision fields for:
  - surface queries
  - free-space queries
  - unknown/ignore queries
  - aggregated query arrays and ignore masks
  - camera/lidar support counts
  - visibility/free-space/unknown fractions
  - intrinsic difficulty target and difficulty component breakdown
- Built a real debug v2 patch cache with 1 sequence and 1 tile:
  - `outputs/ss3dm_prior/teacher_patch_cache_v2_debug`
- Confirmed real checker PNG output:
  - `outputs/ss3dm_prior/teacher_patch_cache_v2_debug/visualizations_v2/Town01__1000_streetsurf__tile_000000.png`
- Real debug patch stats from checker:
  - `visible_surface_fraction_mean: 0.1133`
  - `free_space_fraction_mean: 0.4000`
  - `unknown_fraction_mean: 0.2000`
  - `intrinsic_patch_difficulty_mean: 0.4060`

### Risk / TODO

- `camera_support_count` is currently a pose-proximity support signal, not a full camera visibility proof; this is acceptable for Step 2 but may need stronger camera-based filtering later.
- The current free-space sampler uses LiDAR segment-patch intersection and random sampling inside valid ray intervals; later steps may want denser or stratified sampling for implicit-field supervision.
- Real v2 cache generation now reaches back into raw LiDAR frames, so large-scale builds will be costlier than the old v1 cache build and may need additional caching or worker tuning.

## Step 3 - Hybrid V2 Model, Prototype Memory, Occupancy Head, and Difficulty Losses

### Modified Files

- `ss3dm_prior/models/__init__.py`
- `ss3dm_prior/models/vector_quantizer.py`
- `ss3dm_prior/models/hybrid_patch_prior_v2.py`
- `ss3dm_prior/models/patch_denoiser.py`
- `ss3dm_prior/losses.py`
- `ss3dm_prior/metrics.py`
- `ss3dm_prior/data/train_dataset.py`
- `configs/ss3dm_prior/model_v8_hybrid.yaml`
- `tests/ss3dm_prior/test_model_v2_forward.py`
- `docs/ss3dm_prior_v2_plan.md`
- `docs/ss3dm_prior_v2_update_log.md`
- `docs/ss3dm_prior_v2_experiments.md`

### Design Rationale

- Preserve the trainer import path by keeping `LocalPatchDenoiser` as the public entrypoint and dispatch internally by `model_type`.
- Keep `legacy_v1` untouched in behavior so older configs still instantiate the original reconstruction-and-score model.
- Add a parallel `hybrid_v2` implementation with vector-quantized prototype memory and an occupancy head, but without wiring new trainer logic yet.
- Make dataset/loss/metric logic opportunistic: when v2 fields exist they are used, otherwise they degrade to zero-loss/no-op behavior to stay backward compatible.

### Actual Commands

- `pytest tests/ss3dm_prior/test_model_v2_forward.py -q`
- `pytest tests/ss3dm_prior/test_model_forward.py -q`
- `python - <<'PY' ... real v2 patch batch forward ... PY`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/models/__init__.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/vector_quantizer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/hybrid_patch_prior_v2.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/patch_denoiser.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/losses.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/metrics.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/data/train_dataset.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_model_v2_forward.py"`

### Result

- Added a standard VQ codebook module with:
  - `quantized_latent`
  - `code_indices`
  - `vq_commitment_loss`
  - `codebook_stats`
- Added `HybridPatchPriorV2` with:
  - corrupted encoder
  - observed encoder
  - optional clean teacher encoder
  - fused latent projection
  - vector-quantized prototype memory
  - point decoder
  - implicit occupancy head
  - point defect head
  - legacy corruption-score head
  - intrinsic difficulty head
  - prototype/retrieval embedding head
- Added `model_type` switch support:
  - `legacy_v1`
  - `hybrid_v2`
- Added hybrid config:
  - `configs/ss3dm_prior/model_v8_hybrid.yaml`
- Extended losses with:
  - `corruption_score_loss`
  - `intrinsic_difficulty_loss`
  - `intrinsic_difficulty_pairwise_loss`
  - `occupancy_bce_loss`
  - `free_space_violation_loss`
  - `vq_commitment_loss`
  - `prototype_diversity_loss`
- Extended metrics with:
  - `occupancy_iou_visible`
  - `free_space_violation_rate`
  - `intrinsic_difficulty_mae`
  - `intrinsic_difficulty_spearman`
  - `prototype_usage_entropy`
- Extended `TeacherPatchTrainDataset` to expose v2 patch fields when present.
- Real v2 patch batch forward succeeded with:
  - `patch_id Town01__1000_streetsurf__tile_000000`
  - `query_points_all (1, 1280, 3)`
  - `query_occupancy_logits (1, 1280)`
  - `intrinsic_difficulty_pred (1,)`
  - `total_loss 0.9140893220901489`

### Risk / TODO

- Trainer still does not collate or log the new v2 batch fields, which is intentional for this step but means full training integration is still pending.
- `intrinsic_difficulty_spearman` inherits the existing constant-input behavior and can return `NaN` on tiny or degenerate batches.
- The current occupancy head uses only query coordinates plus quantized global latent; later steps may want stronger local conditioning or visibility-aware features.

## Step 4 - Trainer Upgrade, Curriculum, Checkpoints, and V2 W&B Logging

### Modified Files

- `ss3dm_prior/engine/trainer.py`
- `ss3dm_prior/train.py`
- `ss3dm_prior/viz/render_patch_panels.py`
- `ss3dm_prior/viz/render_sequence_maps.py`
- `configs/ss3dm_prior/train_v8_hybrid.yaml`
- `scripts/ss3dm_prior/train_v8_hybrid.sh`
- `tests/ss3dm_prior/test_train_v2_smoke.py`
- `docs/ss3dm_prior_v2_plan.md`
- `docs/ss3dm_prior_v2_update_log.md`
- `docs/ss3dm_prior_v2_experiments.md`

### Design Rationale

- Keep one trainer path and one stable training CLI, but add model-aware optional logic so `legacy_v1` does not break while `hybrid_v2` can consume new supervision.
- Use config-based curriculum by zeroing selected loss weights before their stage start epochs. This keeps the train loop simple and avoids optimizer restarts.
- Emphasize hard patches using weighted sampling from `intrinsic_patch_difficulty_target`, which is safer than dropping easy patches entirely.
- Keep all qualitative outputs as static PNGs so they remain compatible with headless environments and `wandb.Image`.

### Actual Commands

- `python -m ss3dm_prior.train --help`
- `pytest tests/ss3dm_prior/test_train_v2_smoke.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/train.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/viz/render_patch_panels.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/viz/render_sequence_maps.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_train_v2_smoke.py"`
- `python - <<'PY' ... one-epoch real hybrid debug training ... PY`

### Result

- Trainer now collates and forwards v2 tensors such as:
  - `query_points_all`
  - `query_labels_all`
  - `query_ignore_mask`
  - `intrinsic_patch_difficulty_target`
- Added curriculum-aware loss scheduling for:
  - occupancy/free-space
  - intrinsic difficulty
  - VQ commitment
  - prototype diversity
- Added hard-example weighted sampling from intrinsic difficulty metadata.
- Added checkpoint selection for:
  - `best_recon.pt`
  - `best_gain.pt`
  - `best_composite.pt`
  - `best_visibility.pt`
- Added validation metrics/logging for:
  - `val_intrinsic_difficulty_mae`
  - `val_intrinsic_difficulty_spearman`
  - `val_occupancy_iou_visible`
  - `val_free_space_violation_rate`
  - `val_prototype_usage_entropy`
  - `val_retrieval_top1_nonself`
  - `val_retrieval_top5_nonself`
- Added new qualitative PNGs and W&B image logging for:
  - `visibility_panel`
  - `hybrid_reconstruction_panel`
  - `sequence_visibility_map`
  - `prototype_usage_gallery`
- Real 1-epoch hybrid debug training succeeded at:
  - `outputs/ss3dm_prior/train_v8_hybrid_debug_step4`
- Real persistent run produced 8 PNGs, including at least these new v2 visuals:
  - `Town01__1000_streetsurf__tile_000000_visibility_panel.png`
  - `Town01__1000_streetsurf__tile_000000_hybrid_reconstruction_panel.png`
  - `Town01__1000_streetsurf_hardest_sequence_visibility_map.png`
  - `prototype_usage_gallery.png`

### Risk / TODO

- Tiny validation sets still make several rank/retrieval metrics undefined, so `NaN` can legitimately appear in hybrid debug runs.
- `sequence_visibility_map` now avoids always taking the first sequence, but on single-sequence debug runs only one selection category is available.
- Prototype gallery currently shows first representative patches per prototype id; later steps may want stronger prototype summarization or top-k prototype weighting visualizations.

## Step 5 - Eval / Report Upgrade for Strict Protocol and V2 Metrics

### Modified Files

- `ss3dm_prior/eval.py`
- `ss3dm_prior/reporting.py`
- `scripts/ss3dm_prior/eval_v8_hybrid.sh`
- `tests/ss3dm_prior/test_eval_v2_smoke.py`
- `docs/ss3dm_prior_v2_experiments.md`
- `docs/ss3dm_prior_v2_update_log.md`

### Design Rationale

- Keep one stable eval CLI, but extend it so the same checkpoint can now emit strict-protocol context, visibility/free-space behavior, intrinsic difficulty quality, non-self retrieval, and prototype usage in one pass.
- Reuse the cached v2 patch fields exposed by `TeacherPatchTrainDataset` instead of rebuilding visibility supervision during eval.
- Preserve the legacy self-aligned retrieval metric in the report header for comparison, while making the non-self metrics equally prominent so the old metric no longer dominates interpretation.
- Shift qualitative output selection from generic best/worst gain examples to failure-mode-oriented views: hidden completion, free-space violations, intrinsic score error, prototype gallery, and sequence-level visibility maps.

### Actual Commands

- `python -m ss3dm_prior.eval --help`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/eval.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/reporting.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_eval_v2_smoke.py"`
- `pytest tests/ss3dm_prior/test_eval_v2_smoke.py -q`
- `python -m ss3dm_prior.eval --checkpoint "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_v8_hybrid_debug_step4/checkpoints/best_composite.pt" --manifest_path "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" --patch_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v2_debug" --split_config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/partial_town01_eval.yaml" --output_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior_eval" --eval_name "step5_debug_eval" --wandb_mode disabled`

### Result

- `metrics_summary.json` now exports these top-level eval fields:
  - `recon_chamfer_l1`
  - `recon_normal_cosine`
  - `denoise_gain_chamfer`
  - `intrinsic_difficulty_mae`
  - `intrinsic_difficulty_spearman`
  - `occupancy_iou_visible`
  - `free_space_violation_rate`
  - `retrieval_top1_self_aligned`
  - `retrieval_top1_nonself`
  - `retrieval_top5_nonself`
  - `prototype_usage_entropy`
  - `protocol_valid`
- Per-town and per-sequence CSV outputs now include the v2-aggregable subset plus:
  - patch count
  - sequence count where applicable
  - mean visible support
  - mean intrinsic difficulty
  - mean denoise gain
  - prototype usage entropy
- `report.md` now starts with:
  - checkpoint path
  - split config path and split town summary
  - protocol audit result
  - strict validity
  - legacy vs non-self retrieval comparison
  - conclusion template
- Added new eval qualitative outputs:
  - `best_hidden_completion`
  - `worst_free_space_violation`
  - `largest_intrinsic_score_error`
  - `prototype_gallery`
  - `sequence_visibility_map`
- Added hybrid eval launcher:
  - `scripts/ss3dm_prior/eval_v8_hybrid.sh`
- New smoke test `tests/ss3dm_prior/test_eval_v2_smoke.py` trains a tiny strict-valid hybrid checkpoint, evaluates it on a held-out town with three sequences, and verifies the new summary/report/PNG outputs.
- Real debug eval output was written to:
  - `outputs/ss3dm_prior_eval/step5_debug_eval`
- Real debug eval confirmed new qualitative files:
  - `patch_panels/best_hidden_completion__Town01__1000_streetsurf__tile_000000.png`
  - `patch_panels/worst_free_space_violation__Town01__1000_streetsurf__tile_000000.png`
  - `patch_panels/largest_intrinsic_score_error__Town01__1000_streetsurf__tile_000000.png`
  - `prototype_gallery/prototype_gallery.png`
  - `sequence_maps/hardest__Town01__1000_streetsurf.png`
- Real debug eval also surfaced the expected protocol warnings from the Step 4 debug run:
  - `debug_use_all_patches_for_train_val=true`
  - `allow_debug_split_override=true`
  - `allow_split_fallback=true`
  - overlapping train/val towns in the saved run split

### Risk / TODO

- Tiny eval banks can still make `retrieval_top1_nonself`, `retrieval_top5_nonself`, or cross-sequence retrieval undefined; this correctly degrades to warnings plus `NaN`, but larger held-out evals are still needed for meaningful retrieval conclusions.
- The current `best_hidden_completion` ranking is a heuristic based on `unknown_fraction * positive_denoise_gain`; later steps may want a more geometry-grounded hidden-surface completion score.
- Sequence visibility maps now select up to three distinct sequences, but a single-sequence debug cache still yields only one map by construction.

## Step 6 - Paper-Style Ablation Orchestration and Aggregation

### Modified Files

- `ss3dm_prior/models/hybrid_patch_prior_v2.py`
- `ss3dm_prior/tools/run_ablation_suite.py`
- `ss3dm_prior/tools/aggregate_ablation_results.py`
- `scripts/ss3dm_prior/run_ablation_suite.sh`
- `docs/ss3dm_prior_v2_experiments.md`
- `docs/ss3dm_prior_v2_update_log.md`
- `docs/ss3dm_prior_v2_full_progress_report_2026-04-14.md`

### Design Rationale

- Move from single-run checkpoint inspection to a reproducible suite runner that materializes multiple ablation configs, runs train/eval for each, and records the chosen checkpoint plus summary metrics in one manifest.
- Keep orchestration low-coupling by building ablation variants as config overlays on top of the stabilized Step 1-5 configs instead of creating a new training path.
- Add a real `use_vector_quantization` switch to `HybridPatchPriorV2` so `v2_visibility_no_vq` is a true architectural ablation rather than just zeroing VQ losses while still quantizing latents.
- Separate aggregation into its own tool so future reruns can regenerate the paper table from a saved manifest without retraining.

### Actual Commands

- `python -m ss3dm_prior.tools.run_ablation_suite --help`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/models/hybrid_patch_prior_v2.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/run_ablation_suite.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/aggregate_ablation_results.py"`
- `python -m ss3dm_prior.tools.run_ablation_suite --output_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior_ablations" --suite_name "step6_debug_suite" --debug_synthetic --wandb_mode disabled`

### Result

- Added new orchestration tool:
  - `ss3dm_prior.tools.run_ablation_suite`
- Added new aggregation tool:
  - `ss3dm_prior.tools.aggregate_ablation_results`
- Added launcher:
  - `scripts/ss3dm_prior/run_ablation_suite.sh`
- Standard ablation suite now supports:
  - `legacy_v1_strict`
  - `v2_no_visibility`
  - `v2_visibility_no_vq`
  - `v2_visibility_plus_vq`
  - `v2_full`
  - optional `v2_no_camera_visibility` when a lidar-only patch cache is supplied
- The suite writes:
  - `suite_manifest.json`
  - `ablation_summary.csv`
  - `ablation_summary.md`
- Debug suite successfully ran all five required variants at:
  - `outputs/ss3dm_prior_ablations/step6_debug_suite`
- The generated aggregate CSV contains the requested paper-table metrics:
  - `recon_chamfer_l1`
  - `denoise_gain_chamfer`
  - `intrinsic_difficulty_spearman`
  - `occupancy_iou_visible`
  - `free_space_violation_rate`
  - `retrieval_top1_nonself`
  - `protocol_valid`
- Debug suite aggregate snapshot:
  - `legacy_v1_strict`: `recon_chamfer_l1=1.3048`, `denoise_gain_chamfer=0.0438`, `retrieval_top1_nonself=0.1667`, `protocol_valid=True`
  - `v2_no_visibility`: `recon_chamfer_l1=1.3398`, `denoise_gain_chamfer=0.0088`, `intrinsic_difficulty_spearman=0.2571`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.3333`, `protocol_valid=True`
  - `v2_visibility_no_vq`: `recon_chamfer_l1=1.3396`, `denoise_gain_chamfer=0.0090`, `intrinsic_difficulty_spearman=0.0286`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`, `protocol_valid=True`
  - `v2_visibility_plus_vq`: `recon_chamfer_l1=1.3208`, `denoise_gain_chamfer=0.0278`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`, `protocol_valid=True`
  - `v2_full`: `recon_chamfer_l1=1.3208`, `denoise_gain_chamfer=0.0278`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`, `protocol_valid=True`
- Added a consolidated full-progress report with timestamp:
  - `docs/ss3dm_prior_v2_full_progress_report_2026-04-14.md`

### Risk / TODO

- The debug suite uses a synthetic strict-valid patch cache to validate orchestration mechanics; the resulting numbers are only smoke-level and not suitable for paper claims.
- `v2_no_camera_visibility` is implemented as an optional suite member because it requires a separate lidar-only patch cache or equivalent preprocessing choice.
- The current aggregation table records one selected checkpoint per variant; later paper writing may still want an explicit policy note on whether legacy and v2 rows are compared via `best_gain`, `best_recon`, or `best_composite`.
