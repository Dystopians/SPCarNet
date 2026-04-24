# SS3DM Prior V3 Experiments

## Step 1

### Strict Control Baseline

- New baseline config: `configs/ss3dm_prior/train_v9_strict_control.yaml`
- Launch script: `scripts/ss3dm_prior/train_v9_strict_control.sh`
- Intent: keep the existing SS3DM prior model family unchanged while making the control run strict-protocol, auditable, and report-clean.

### Protocol and Reporting Notes

- Eval now exports protocol status directly into `metrics_summary.json` and `report.md`.
- Reported difficulty is split into two distinct axes:
  - `mean_corruption_score_target` for synthetic corruption difficulty
  - `mean_intrinsic_difficulty` for intrinsic geometry difficulty
- Strict/debug/fallback flags are now visible in evaluation outputs instead of being hidden inside training config files.

### Validation Commands

- `python -m ss3dm_prior.train --help`
- `python -m ss3dm_prior.eval --help`
- `python -m ss3dm_prior.tools.audit_run_protocol --help`
- `pytest tests/ss3dm_prior/test_train_entrypoint_wandb.py -q`
- `pytest tests/ss3dm_prior/test_loss_weighting.py -q`
- `python -m py_compile "...modified files for Step 1..."`

### Follow-up for Later Steps

- Add real strict control training/eval results under the new v9 baseline once Step 2 scope is approved.
- Decide whether `grad_accum_steps` should be implemented in trainer or remain a documented baseline TODO.

## Step 2

### V3 Patch Cache

- New config: `configs/ss3dm_prior/teacher_patch_v3.yaml`
- New builder: `ss3dm_prior.tools.build_teacher_patch_cache_v3`
- New checker: `ss3dm_prior.tools.check_teacher_patch_cache_v3`
- New builder module: `ss3dm_prior.data.teacher_patch_builder_v3`

### V3 Semantic Fields

- Multi-scale identifiers:
  - `scale_id`
  - `patch_radius_m`
- Visible / hidden geometry:
  - `visible_clean_points`
  - `visible_clean_normals`
  - `hidden_clean_points`
  - `hidden_clean_normals`
  - `surface_support_mask`
- Free-space hard negatives:
  - `free_space_query_hard_negatives`
  - `free_space_hard_negative_count`

### Multi-Scale Patch ID Rule

- Deterministic rule:
  - `{sequence_id}__tile_{tile_id:06d}__scale_{scale_id:02d}__r{patch_radius_m:.2f}m`
- Radius token uses `.` -> `p`, for example:
  - `Town01__1000_streetsurf__tile_000000__scale_01__r4p00m`

### Corruption Curriculum

- Dataset now supports:
  - `set_epoch(epoch)`
  - epoch-aware `severity_schedule`
- Supported schedule configs:
  - `type: linear | cosine`
  - `start_scale`
  - `end_scale`
  - `warmup_epochs`
- Validation and eval keep fixed corruption and do not consume epoch schedule.

### Eval Additions

- Added visible/hidden geometry reporting fields:
  - `visible_recon_chamfer_l1`
  - `hidden_completion_chamfer_l1`

### Validation Commands

- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v3 --help`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v3 --help`
- `pytest tests/ss3dm_prior/test_corruption_schedule.py -q`
- `pytest tests/ss3dm_prior/test_patch_cache_v3.py -q`
- `python -m py_compile "...modified files for Step 2..."`
- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v3 --manifest "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" --split_config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/default_town_split.yaml" --config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/teacher_patch_v3.yaml" --observed_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache" --town_mesh_cache_root "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache" --out_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v3_debug" --subsets train --debug_max_sequences 1 --debug_max_tiles_per_sequence 1 --num_workers 1 --seed 0`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v3 --patch_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v3_debug" --num_visualizations 2 --seed 0`

### Real Debug Build Observations

- Output root:
  - `outputs/ss3dm_prior/teacher_patch_cache_v3_debug`
- Real built patch count:
  - `2`
- Real built scales:
  - `scale_1`
  - `scale_2`
- Checker summary highlights:
  - `visible_support_fraction_mean: 0.3767`
  - `hidden_surface_fraction_mean: 0.6233`
  - `free_space_hard_negative_count_mean: 96.0000`

### Follow-up for Later Steps

- Scale `0` patches were skipped in the real debug run because the smallest radius did not satisfy current patch validity thresholds; later steps may decide whether that is desirable or whether v3 should relax per-scale thresholds.
- Trainer now exposes corruption schedule state, but no v3-specific model or loss uses visible/hidden supervision yet.

## Step 3

### V9 Wide Baseline

- New model config: `configs/ss3dm_prior/model_v9_wide.yaml`
- New train config: `configs/ss3dm_prior/train_v9_wide.yaml`
- New launch script: `scripts/ss3dm_prior/train_v9_wide.sh`
- New dispatch type: `model_type: hybrid_v2_wide`

### Capacity Upgrade Scope

- No cross-attention
- No diffusion
- No trainer rewrite
- Only wider capacity on the current hybrid path:
  - wider PointNet encoders
  - wider fusion MLP
  - wider reconstruction decoder
  - wider defect / score / intrinsic / retrieval heads
  - wider occupancy head
  - optional residual MLP blocks
  - optional layer norm / dropout

### Default Wide Capacity

- `latent_dim: 512`
- `retrieval_dim: 256`
- `recon_point_count: 3072`
- `codebook_size: 512`
- `occupancy_hidden_dim: 384`
- `teacher_encoder_enabled: true`
- `use_vector_quantization: true`

### Default Wide Loss Weights

- `recon_chamfer_loss: 1.0`
- `recon_normal_loss: 0.3`
- `point_defect_loss: 0.3`
- `corruption_score_loss: 0.05`
- `intrinsic_difficulty_loss: 0.25`
- `occupancy_bce_loss: 0.25`
- `free_space_violation_loss: 0.25`
- `vq_commitment_loss: 0.05`
- `prototype_diversity_loss: 0.01`

### Validation Commands

- `pytest tests/ss3dm_prior/test_model_v9_wide_forward.py -q`
- `python -m py_compile "...modified files for Step 3..."`
- `python - <<'PY' ... real v3 patch wide forward ... PY`

### Forward Validation Outcome

- Random-tensor wide forward passed with:
  - `recon_points: (1, 3072, 3)`
  - `query_occupancy_logits: (1, 64)`
  - `retrieval_embedding: (1, 256)`
- Real v3 patch batch forward passed with:
  - `patch_id: Town01__1000_streetsurf__tile_000000__scale_01__r4p00m`
  - `recon_points: (1, 3072, 3)`
  - `query_occupancy_logits: (1, 1280)`
  - `retrieval_embedding: (1, 256)`

### Follow-up for Later Steps

- Use this wide baseline to answer whether the current bottleneck is under-capacity before introducing any new modeling paradigm.
- If training becomes unstable or memory-bound, revisit width-vs-depth tradeoffs before adding new objectives.

## Step 4

### V10 Cross-Attention Hybrid

- New model config: `configs/ss3dm_prior/model_v10_crossattn.yaml`
- New train config: `configs/ss3dm_prior/train_v10_crossattn.yaml`
- New launch script: `scripts/ss3dm_prior/train_v10_crossattn.sh`
- New dispatch type: `model_type: v10_cross_attention_hybrid`

### Architecture Scope

- No diffusion
- No trainer rewrite
- No full point transformer
- Local-patch latent cross-attention only:
  - corrupted patch encoder -> token set
  - observed patch encoder -> token set
  - optional visible / hidden / query token sets
  - latent query bank with prototype-conditioned refinement

### Default Attention Capacity

- `latent_dim: 512`
- `retrieval_dim: 256`
- `codebook_size: 512`
- `occupancy_hidden_dim: 384`
- `num_latent_queries: 48`
- `num_cross_attention_layers: 4`
- `num_latent_self_attention_layers: 2`
- `attention_heads: 8`
- `ffn_dim: 1024`
- `dropout: 0.1`

### Compatibility Notes

- Output schema stays aligned with existing loss / trainer / eval expectations:
  - `recon_points`
  - `recon_normals`
  - `point_defect_pred`
  - `patch_score_pred`
  - `intrinsic_difficulty_pred`
  - `query_occupancy_logits`
  - `retrieval_embedding`
  - `code_indices`
  - `vq_commitment_loss`
- Trainer and eval now move collated variable-length visible / hidden tensors to device and pass them through to the model when available.

### Parameter Count

- `hybrid_v2_wide`: `40052874`
- `v10_cross_attention_hybrid`: `52579338`

### Validation Commands

- `pytest tests/ss3dm_prior/test_model_v10_crossattn_forward.py -q`
- `python -m py_compile "...modified files for Step 4..."`
- `python - <<'PY' ... print v9/v10 parameter counts ... PY`
- `python - <<'PY' ... real v3 patch cross-attention forward ... PY`

### Forward Validation Outcome

- Random-tensor cross-attention forward passed with:
  - `recon_points: (1, 3072, 3)`
  - `query_occupancy_logits: (1, 80)`
  - `retrieval_embedding: (1, 256)`
- Real v3 patch batch forward passed with:
  - `patch_id: Town01__1000_streetsurf__tile_000000__scale_01__r4p00m`
  - `recon_points: (1, 3072, 3)`
  - `query_occupancy_logits: (1, 1280)`
  - `retrieval_embedding: (1, 256)`
  - `intrinsic_difficulty_pred: (1,)`

### Follow-up for Later Steps

- Run actual training/eval against `hybrid_v2_wide` before concluding whether cross-attention improves geometry completion rather than only capacity.
- If GPU memory becomes limiting, ablate latent query count, decoder width, and token encoder width before changing the overall attention recipe.

## Step 5

### Trainer Stabilization Scope

- New trainer-side features:
  - `grad_accum_steps`
  - `ema.enable`
  - `ema.decay`
  - `ema.use_for_eval`
  - staged `recon_warmup_epochs`
  - existing dataset corruption schedule kept active via `dataset.set_epoch(epoch)`
  - hard-example sampling and hard-example loss weighting supported together
- New checkpoint:
  - `best_paper.pt`

### Default Paper Checkpoint Score

- `val_denoise_gain_chamfer`
- `val_recon_chamfer_l1`
- `val_intrinsic_difficulty_spearman`
- `val_occupancy_iou_visible`
- `val_free_space_violation_rate`
- `val_retrieval_top1_nonself`

### New Metrics Carried Through Trainer / Eval / Report

- `visible_recon_chamfer_l1`
- `hidden_completion_chamfer_l1`
- `visible_recon_normal_cosine`
- `hidden_completion_gain`
- `intrinsic_difficulty_calibration_mae`
- `free_space_fp_rate`
- `retrieval_top1_nonself`
- `retrieval_top5_nonself`

### New Qualitative Artifacts

- `visible_vs_hidden_panel`
- `free_space_error_panel`
- `difficulty_calibration_panel`
- `prototype_usage_gallery`
- hardest / best_gain / worst_gain `sequence_visibility_map`

### Ablation Suite Variants

- `v8_strict_control`
- `v9_wide`
- `v10_crossattn`
- `v10_crossattn_multiscale`
- `v10_crossattn_multiscale_stronger_curriculum`

### Ablation Outputs

- `suite_manifest.json`
- `ablation_summary.csv`
- `ablation_summary.md`

### Validation Commands

- `pytest tests/ss3dm_prior/test_trainer_v3_smoke.py -q`
- `pytest tests/ss3dm_prior/test_eval_v3_smoke.py -q`
- `python -m py_compile "...modified files for Step 5..."`

### Validation Status In This Session

- IDE lint diagnostics on the modified Step 5 files were clean.
- Shell-backed command execution in this agent session was not reliable, so the Step 5 `pytest` and `py_compile` commands were attempted but not trustworthily observed.
- To compensate, Step 5 now includes:
  - a synthetic v3 trainer smoke test
  - a synthetic v3 eval smoke test
  - synthetic v3 fixture generation for new PNG classes and new metric keys

### Follow-up for Later Steps

- Rerun the Step 5 smoke tests and `py_compile` in a healthy terminal session before treating the ablation suite as fully verified.
- Once shell execution is healthy, run the full `run_v3_ablation_suite.sh` against the real baseline and multiscale caches to populate the paper table artifacts.

## Step 6

### V11 Latent Flow Exploratory Branch

- New model config: `configs/ss3dm_prior/model_v11_latent_flow.yaml`
- New train config: `configs/ss3dm_prior/train_v11_latent_flow.yaml`
- New launch script: `scripts/ss3dm_prior/train_v11_latent_flow.sh`
- New dispatch type: `model_type: v11_latent_flow_hybrid`

### Exploratory Scope

- No raw point diffusion
- No replacement of deterministic `v10`
- Latent flow matching on top of the existing conditional cross-attention backbone
- Hidden-region residual completion, with clean latent residual fallback when hidden supervision is missing

### Conditioning and Reranking

- Flow condition includes:
  - corrupted latent summary
  - observed latent summary
  - visible summary
  - hidden summary
  - prototype-conditioned latent
- Eval reranking uses:
  - observed consistency
  - visible consistency
  - free-space violation penalty
  - prototype consistency

### New Eval Outputs

- `best_of_k_hidden_completion`
- `mean_of_k_hidden_completion`
- `sample_diversity`
- `free_space_safe_best_of_k`
- deterministic vs stochastic comparison table for `K=1/4/8`
- qualitative `*_stochastic_candidates.png` galleries

### Validation Commands

- `pytest tests/ss3dm_prior/test_model_v11_latent_flow_forward.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/models/latent_flow_patch_prior_v11.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/patch_denoiser.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/losses.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/eval.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/reporting.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/viz/render_patch_panels.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_model_v11_latent_flow_forward.py"`

### Validation Outcome

- `pytest tests/ss3dm_prior/test_model_v11_latent_flow_forward.py -q`: passed (`3 passed`)
- `python -m py_compile ...`: passed
- Tiny eval now writes stochastic comparison metrics and stochastic candidate galleries.

### Follow-up for Later Steps

- Keep `v11` exploratory until it is compared against deterministic `v10` on real validation/test runs rather than only smoke fixtures.
- If candidate generation is too slow, first ablate flow steps and `K` before changing the backbone or switching to a heavier diffusion recipe.
