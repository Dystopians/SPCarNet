# SS3DM Prior V2 Experiments

## Step 1

### Control Baseline

- Baseline config for future v2 ablations: `configs/ss3dm_prior/train_v8_strict.yaml`
- Companion launch script: `scripts/ss3dm_prior/train_v8_strict.sh`
- Intent: preserve the current v1 model family while enforcing strict split protocol flags

### Protocol Audit Notes

- Audited run: `outputs/ss3dm_prior/train_full_v5`
- Audit outcome: `protocol_valid=false`
- Observed warnings:
  - debug leakage via `debug_use_all_patches_for_train_val=true`
  - `allow_split_fallback=true`
  - `allow_debug_split_override` missing from saved metadata, so legacy runs may be only partially auditable

### Metric Notes

- Legacy metrics retained and renamed:
  - `retrieval_top1_self_aligned`
  - `retrieval_top5_self_aligned`
- New Step 1 filtered metrics:
  - `retrieval_top1_nonself`
  - `retrieval_top5_nonself`
  - `retrieval_top1_cross_sequence`
- Step 1 non-self retrieval is defined as clean-neighbor agreement under a bank that excludes the query `patch_id`. Cross-sequence further restricts candidates to different `sequence_id` values.

### Validation Commands

- `python -m ss3dm_prior.tools.audit_run_protocol --help`
- `python -m ss3dm_prior.eval --help`
- `python -m ss3dm_prior.tools.audit_run_protocol "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_full_v5"`
- `pytest tests/ss3dm_prior/test_protocol_audit.py -q`
- `pytest tests/ss3dm_prior/test_eval_smoke.py -q`

### Follow-up for Later Steps

- Run a full `train_v8_strict` control training job and store protocol-audited eval results before introducing visibility-aware retrieval or new v2 model blocks.
- Decide whether a later step should add a geometry-grounded non-self retrieval benchmark in addition to the current embedding-neighborhood agreement metric.

## Step 2

### Cache V2 Output

- New config: `configs/ss3dm_prior/teacher_patch_v2.yaml`
- New builder: `ss3dm_prior.tools.build_teacher_patch_cache_v2`
- New checker: `ss3dm_prior.tools.check_teacher_patch_cache_v2`
- Real debug output root:
  - `outputs/ss3dm_prior/teacher_patch_cache_v2_debug`

### Cached V2 Fields

- Query supervision:
  - `surface_query_points`
  - `surface_query_labels`
  - `free_query_points`
  - `free_query_labels`
  - `unknown_query_points`
  - `query_points_all`
  - `query_labels_all`
  - `query_ignore_mask`
- Support / visibility:
  - `camera_support_count`
  - `lidar_support_count`
  - `visible_surface_fraction`
  - `free_space_fraction`
  - `unknown_fraction`
- Difficulty:
  - `intrinsic_patch_difficulty_target`
  - `difficulty_components_json`

### Debug Build Commands

- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v2 --manifest "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" --split_config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/default_town_split.yaml" --config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/teacher_patch_v2.yaml" --observed_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache" --town_mesh_cache_root "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache" --out_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v2_debug" --subsets train --debug_max_sequences 1 --debug_max_tiles_per_sequence 1 --num_workers 1 --seed 0`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v2 --patch_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v2_debug" --num_visualizations 1 --seed 0`

### Debug Build Observations

- Built patch count: `1`
- Sequence used: `Town01__1000_streetsurf`
- Confirmed checker PNG:
  - `outputs/ss3dm_prior/teacher_patch_cache_v2_debug/visualizations_v2/Town01__1000_streetsurf__tile_000000.png`
- Queried patch field sizes:
  - `surface_query_points: (512, 3)`
  - `free_query_points: (512, 3)`
  - `unknown_query_points: (256, 3)`
  - `query_points_all: (1280, 3)`

### Validation Commands

- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v2 --help`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v2 --help`
- `pytest tests/ss3dm_prior/test_visibility_queries.py -q`
- `pytest tests/ss3dm_prior/test_teacher_patch_builder.py -q`

### Follow-up for Later Steps

- Use the cached v2 query fields in later model/data loaders instead of regenerating occupancy supervision online.
- Revisit whether camera depth/mask signals are stable enough to upgrade `camera_support_count` into true camera-derived free-space supervision.

## Step 3

### Hybrid Model Config

- New hybrid config: `configs/ss3dm_prior/model_v8_hybrid.yaml`
- Stable model switch:
  - `model_type: legacy_v1`
  - `model_type: hybrid_v2`

### Hybrid V2 Outputs

- Reconstruction:
  - `recon_points`
  - `recon_normals`
- Prototype memory:
  - `quantized_latent`
  - `code_indices`
  - `vq_commitment_loss`
  - `codebook_stats`
- Occupancy:
  - `query_occupancy_logits`
- Prediction heads:
  - `point_defect_pred`
  - `patch_score_pred`
  - `intrinsic_difficulty_pred`
  - `retrieval_embedding`

### Validation Commands

- `pytest tests/ss3dm_prior/test_model_v2_forward.py -q`
- `pytest tests/ss3dm_prior/test_model_forward.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/models/__init__.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/vector_quantizer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/hybrid_patch_prior_v2.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/patch_denoiser.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/losses.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/metrics.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/data/train_dataset.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_model_v2_forward.py"`

### Real V2 Patch Forward Check

- Source patch cache:
  - `outputs/ss3dm_prior/teacher_patch_cache_v2_debug/patch_index.jsonl`
- Forward check result:
  - `patch_id Town01__1000_streetsurf__tile_000000`
  - `query_points_all (1, 1280, 3)`
  - `recon_points (1, 2048, 3)`
  - `query_occupancy_logits (1, 1280)`
  - `intrinsic_difficulty_pred (1,)`
  - `code_indices (1,)`
  - `total_loss 0.9140893220901489`

### Step 3 Metric Additions

- `occupancy_iou_visible`
- `free_space_violation_rate`
- `intrinsic_difficulty_mae`
- `intrinsic_difficulty_spearman`
- `prototype_usage_entropy`

### Follow-up for Later Steps

- Wire v2 query and difficulty tensors through trainer collation and validation logging.
- Decide whether prototype memory should stay standard VQ or switch to EMA VQ after initial training experiments.

## Step 4

### Hybrid Train Config

- New train config: `configs/ss3dm_prior/train_v8_hybrid.yaml`
- New train script: `scripts/ss3dm_prior/train_v8_hybrid.sh`

### Trainer Additions

- Dual-model compatibility:
  - `legacy_v1`
  - `hybrid_v2`
- Curriculum stages:
  - warmup
  - transition
  - main
- Hard-example weighted sampling based on `intrinsic_patch_difficulty_target`
- New checkpoints:
  - `best_composite.pt`
  - `best_visibility.pt`

### New Validation Metrics

- `val_intrinsic_difficulty_mae`
- `val_intrinsic_difficulty_spearman`
- `val_occupancy_iou_visible`
- `val_free_space_violation_rate`
- `val_prototype_usage_entropy`
- `val_retrieval_top1_nonself`
- `val_retrieval_top5_nonself`

### New Qualitative Outputs

- `visibility_panel`
- `hybrid_reconstruction_panel`
- `sequence_visibility_map`
- `prototype_usage_gallery`

### Validation Commands

- `python -m ss3dm_prior.train --help`
- `pytest tests/ss3dm_prior/test_train_v2_smoke.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/train.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/viz/render_patch_panels.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/viz/render_sequence_maps.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_train_v2_smoke.py"`

### Real One-Epoch Debug Train

- Output root:
  - `outputs/ss3dm_prior/train_v8_hybrid_debug_step4`
- Observed output PNGs:
  - `Town01__1000_streetsurf__tile_000000_visibility_panel.png`
  - `Town01__1000_streetsurf__tile_000000_hybrid_reconstruction_panel.png`
  - `Town01__1000_streetsurf_hardest_sequence_gain_map.png`
  - `Town01__1000_streetsurf_hardest_sequence_visibility_map.png`
  - `prototype_usage_gallery.png`

### Follow-up for Later Steps

- Upgrade prototype usage visualization from representative-patch snapshots to top-k prototype weighting summaries if the model later exposes mixture weights.

## Step 5

### Eval Outputs

- New eval launcher: `scripts/ss3dm_prior/eval_v8_hybrid.sh`
- Eval summary now reports:
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

### Grouped Reports

- `metrics_per_town.csv` now includes:
  - patch count
  - sequence count
  - mean visible support
  - mean intrinsic difficulty
  - mean denoise gain
  - prototype usage entropy
- `metrics_per_sequence.csv` now includes:
  - patch count
  - mean visible support
  - mean intrinsic difficulty
  - mean denoise gain
  - prototype usage entropy

### New Qualitative Eval Figures

- `best_hidden_completion`
- `worst_free_space_violation`
- `largest_intrinsic_score_error`
- `prototype_gallery`
- `sequence_visibility_map`

### Validation Commands

- `python -m ss3dm_prior.eval --help`
- `pytest tests/ss3dm_prior/test_eval_v2_smoke.py -q`
- `python -m ss3dm_prior.eval --checkpoint "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_v8_hybrid_debug_step4/checkpoints/best_composite.pt" --manifest_path "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" --patch_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v2_debug" --split_config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/partial_town01_eval.yaml" --output_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior_eval" --eval_name "step5_debug_eval" --wandb_mode disabled`

### Observed Debug Eval Output

- Output root:
  - `outputs/ss3dm_prior_eval/step5_debug_eval`
- Confirmed new files:
  - `patch_panels/best_hidden_completion__Town01__1000_streetsurf__tile_000000.png`
  - `patch_panels/worst_free_space_violation__Town01__1000_streetsurf__tile_000000.png`
  - `patch_panels/largest_intrinsic_score_error__Town01__1000_streetsurf__tile_000000.png`
  - `prototype_gallery/prototype_gallery.png`
  - `sequence_maps/hardest__Town01__1000_streetsurf.png`
- Confirmed summary/report behavior:
  - `metrics_summary.json` includes `protocol_valid=false` for the Step 4 debug checkpoint and surfaces the saved split-protocol warnings.
  - `report.md` header now exposes checkpoint path, split config, strict validity, and legacy-vs-nonself retrieval comparison.

### Follow-up for Later Steps

- Run the new eval script on a strict-valid multi-sequence checkpoint so non-self retrieval and sequence visibility maps become informative beyond the tiny debug cache.
- Consider replacing the current hidden-completion ranking heuristic with a stronger visibility-aware completion score once a later step defines it explicitly.
- Decide whether validation should report sequence-level visibility summaries in CSV/JSON, not only as PNGs and W&B images.

## Step 6

### Ablation Orchestration

- New suite runner: `ss3dm_prior.tools.run_ablation_suite`
- New result aggregator: `ss3dm_prior.tools.aggregate_ablation_results`
- New launcher: `scripts/ss3dm_prior/run_ablation_suite.sh`
- Added a clean `use_vector_quantization` config switch in `HybridPatchPriorV2` so `v2_visibility_no_vq` is a real no-VQ ablation instead of just zeroing VQ losses.

### Standard Ablation Variants

- `legacy_v1_strict`
- `v2_no_visibility`
- `v2_visibility_no_vq`
- `v2_visibility_plus_vq`
- `v2_full`
- Optional if a lidar-only cache is supplied:
  - `v2_no_camera_visibility`

### Aggregated Outputs

- `ablation_summary.csv`
- `ablation_summary.md`
- `suite_manifest.json`

### Validation Commands

- `python -m ss3dm_prior.tools.run_ablation_suite --help`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/models/hybrid_patch_prior_v2.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/run_ablation_suite.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/aggregate_ablation_results.py"`
- `python -m ss3dm_prior.tools.run_ablation_suite --output_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior_ablations" --suite_name "step6_debug_suite" --debug_synthetic --wandb_mode disabled`

### Debug Suite Result

- Output root:
  - `outputs/ss3dm_prior_ablations/step6_debug_suite`
- Generated aggregate files:
  - `ablation_summary.csv`
  - `ablation_summary.md`
  - `suite_manifest.json`
- All five required variants completed with `protocol_valid=True` on the synthetic strict split.
- Debug aggregate metrics:
  - `legacy_v1_strict`: `recon_chamfer_l1=1.3048`, `denoise_gain_chamfer=0.0438`, `retrieval_top1_nonself=0.1667`
  - `v2_no_visibility`: `recon_chamfer_l1=1.3398`, `denoise_gain_chamfer=0.0088`, `intrinsic_difficulty_spearman=0.2571`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.3333`
  - `v2_visibility_no_vq`: `recon_chamfer_l1=1.3396`, `denoise_gain_chamfer=0.0090`, `intrinsic_difficulty_spearman=0.0286`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`
  - `v2_visibility_plus_vq`: `recon_chamfer_l1=1.3208`, `denoise_gain_chamfer=0.0278`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`
  - `v2_full`: `recon_chamfer_l1=1.3208`, `denoise_gain_chamfer=0.0278`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`

### Follow-up for Later Steps

- Replace the synthetic debug suite with a strict-valid multi-town real patch cache before drawing paper conclusions from the ablation table.
- Decide whether the final paper table should compare `best_composite` checkpoints for all v2 variants or fix a single checkpoint-selection policy across all variants, including legacy.
