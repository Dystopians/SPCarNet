# SS3DM Prior Handoff Report

This document records the current factual state of the `ss3dm_prior` work in `mesh-splatting`. It is intended as a handoff artifact for external analysis. It does not contain recommendations, prescriptions, or interpretive conclusions.

## Scope

- Repository: `/data2/peilincai/mesh-splatting`
- Primary package: `ss3dm_prior/`
- Raw dataset root: `/data2/peilincai/SS3DM_raw`
- Current default split file: `configs/ss3dm_prior/splits/default_town_split.yaml`

## Dataset Facts

- Dataset root used by this work: `/data2/peilincai/SS3DM_raw`
- Raw manifest unit: sequence
- Raw towns present in scope: `Town01` to `Town07`, `Town10`
- Default split:
  - train towns: `Town01`, `Town02`, `Town03`, `Town04`, `Town05`, `Town06`
  - val towns: `Town07`
  - test towns: `Town10`
- Split policy fields in `configs/ss3dm_prior/splits/default_town_split.yaml`:
  - `strategy: town_holdout`
  - `unit_of_split: sequence`
  - `forbid_random_patch_split: true`
  - `forbid_random_frame_split: true`

## Implemented Pipeline Stages

Implemented code and docs exist for:

1. Sequence discovery / manifest generation / split skeleton
2. Scenario parsing and sequence-level observed cache generation
3. Town OBJ to binary mesh cache conversion
4. Teacher patch cache construction
5. Online corruption, dataset, model, losses, metrics
6. Training loop, checkpoints, wandb logging, qualitative visualization
7. Standalone evaluation, exported reports, JSON/CSV, qualitative outputs

Existing step-by-step logs and earlier documents:

- `docs/ss3dm_prior_update_log.md`
- `docs/ss3dm_prior_experiments.md`
- `docs/ss3dm_prior_plan.md`
- `docs/ss3dm_prior_data_schema.md`

## Key Post-Step-7 Code Changes

The following files were modified or added after the original Step 7 implementation:

- `ss3dm_prior/models/patch_denoiser.py`
- `ss3dm_prior/losses.py`
- `ss3dm_prior/engine/trainer.py`
- `ss3dm_prior/eval.py`
- `ss3dm_prior/data/train_dataset.py`
- `configs/ss3dm_prior/model_default.yaml`
- `configs/ss3dm_prior/train_default.yaml`
- `configs/ss3dm_prior/model_large_gain.yaml`
- `configs/ss3dm_prior/train_gain_focused.yaml`
- `configs/ss3dm_prior/model_v5_gain.yaml`
- `configs/ss3dm_prior/train_v5_gain.yaml`
- `configs/ss3dm_prior/train_v6_full_strict.yaml`
- `configs/ss3dm_prior/splits/partial_town01_eval.yaml`
- `tests/ss3dm_prior/test_model_forward.py`
- `tests/ss3dm_prior/test_train_dataset.py`
- `tests/ss3dm_prior/test_train_smoke.py`
- `tests/ss3dm_prior/test_eval_smoke.py`

## Model / Loss / Training Changes Applied After Initial v1

### Model-side updates

- Added local canonical frame normalization in `ss3dm_prior/models/patch_denoiser.py`
- Added residual reconstruction mode in `ss3dm_prior/models/patch_denoiser.py`
- Added clean retrieval embedding path in `ss3dm_prior/models/patch_denoiser.py`
- Changed residual decoding from global latent-only output to per-point residual decoding in `ss3dm_prior/models/patch_denoiser.py`
- Added nearest-observed local context features to:
  - residual decoder input
  - point defect head input

### Loss-side updates

- Changed latent alignment from normalized MSE style to cosine-style alignment
- Added `retrieval_align_loss`
- Added per-sample Chamfer and per-sample normal loss support
- Added `hard_example_reweight` in `ss3dm_prior/losses.py`

### Dataset / corruption updates

- Added dynamic corruption visit counting in `ss3dm_prior/data/train_dataset.py`
- Training dataset now uses changing corruption keys across repeated accesses
- Validation / eval datasets can freeze corruption deterministically via `dynamic_corruption=False`

### Trainer / evaluation updates

- Added `train_denoise_gain_chamfer` logging
- Added `retrieval_align_loss` logging
- Added cosine LR scheduler support
- Added gradient clipping support
- Changed retrieval metrics to use retrieval embeddings rather than untrained latent tensors
- Added strict guard against using `debug_use_all_patches_for_train_val: true` on formal split configs unless `allow_debug_split_override: true`

## Model Variants And Parameter Counts

Measured parameter counts:

- `model_default-ish`
  - configuration: `latent_dim=128`, `retrieval_dim=64`
  - parameter count: `3,696,840`

- `model_large_gain`
  - configuration: `latent_dim=256`, `retrieval_dim=128`
  - parameter count: `8,272,776`

- `model_v5_gain`
  - configuration: `latent_dim=256`, `retrieval_dim=96`
  - parameter count: `8,264,552`

## Configuration Files In Current Use

### `configs/ss3dm_prior/model_v5_gain.yaml`

Core facts:

- `latent_dim: 256`
- `retrieval_dim: 96`
- `recon_point_count: 2048`
- `use_observed_condition: true`
- `use_local_frame: true`
- `use_residual_reconstruction: true`

Corruption settings:

- `point_dropout.dropout_ratio: 0.18`
- `gaussian_jitter.sigma: 0.035`
- `normal_noise.sigma: 0.12`
- `local_hole_mask.hole_radius: 0.28`
- `outlier_cluster.cluster_size: 40`
- `density_imbalance.region_radius: 0.32`

Loss weights:

- `recon_chamfer_loss: 1.0`
- `recon_normal_loss: 0.4`
- `point_defect_loss: 0.35`
- `patch_score_loss: 0.1`
- `latent_align_loss: 0.01`
- `retrieval_align_loss: 0.02`
- `hard_example_reweight: 0.6`

### `configs/ss3dm_prior/train_v5_gain.yaml`

Core facts:

- `epochs: 24`
- `batch_size: 16`
- `lr: 0.0004`
- `min_lr: 0.00003`
- `lr_scheduler: cosine`
- `weight_decay: 0.0005`
- `grad_clip_norm: 1.0`
- `step_visualization_interval_steps: 2000`
- `debug_use_all_patches_for_train_val: true`
- `allow_debug_split_override: true`
- `allow_split_fallback: true`

### `configs/ss3dm_prior/train_v6_full_strict.yaml`

Core facts:

- `epochs: 24`
- `batch_size: 16`
- `lr: 0.0004`
- `min_lr: 0.00003`
- `lr_scheduler: cosine`
- `weight_decay: 0.0005`
- `grad_clip_norm: 1.0`
- `step_visualization_interval_steps: 2000`
- `debug_use_all_patches_for_train_val: false`
- `allow_split_fallback: false`

## Patch Cache State

### Partial patch cache

Path:

- `outputs/ss3dm_prior/teacher_patch_cache_partial_try_v2`

Observed coverage from direct inspection:

- towns present: `Town01`
- sequences present:
  - `Town01__1000_streetsurf`
  - `Town01__150_streetsurf`
  - `Town01__300_streetsurf`

Patch counts from inspection:

- total patches in partial cache index coverage check: `11001`

### Full patch cache

Path:

- `outputs/ss3dm_prior/teacher_patch_cache`

Current facts after index rebuild:

- `patch_index.jsonl` line count: `104928`
- `.npz` patch file count: `104928`
- `sequence_patch_stats.json` count: `28`

Full cache sequence directories observed:

- `Town01__1000_streetsurf`
- `Town01__150_streetsurf`
- `Town01__300_streetsurf`
- `Town01__550_streetsurf`
- `Town02__150_streetsurf`
- `Town02__260_streetsurf`
- `Town02__600_streetsurf`
- `Town03__1000_streetsurf`
- `Town03__150_streetsurf`
- `Town03__300_streetsurf`
- `Town03__360_streetsurf`
- `Town03__600_streetsurf`
- `Town04__150_streetsurf`
- `Town04__600_streetsurf`
- `Town05__1000_streetsurf`
- `Town05__150_streetsurf`
- `Town05__300_streetsurf`
- `Town05__600_streetsurf`
- `Town06__150_streetsurf`
- `Town06__990_streetsurf`
- `Town07__1000_streetsurf`
- `Town07__150_streetsurf`
- `Town07__300_streetsurf`
- `Town07__600_streetsurf`
- `Town10__1000_streetsurf`
- `Town10__200_streetsurf`
- `Town10__300_streetsurf`
- `Town10__580_streetsurf`

## Commands Executed During Patch Cache Build / Rebuild

Observed commands executed in terminal history:

```bash
python -m ss3dm_prior.tools.build_teacher_patch_cache \
  --manifest /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json \
  --split_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/default_town_split.yaml \
  --config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/teacher_patch_default.yaml \
  --observed_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache \
  --town_mesh_cache_root /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache \
  --out_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache \
  --subsets train val test \
  --seed 0 \
  --num_workers 12 \
  --skip_completed_sequences
```

```bash
python -m ss3dm_prior.tools.build_teacher_patch_cache \
  --manifest /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json \
  --split_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/default_town_split.yaml \
  --config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/teacher_patch_default.yaml \
  --observed_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache \
  --town_mesh_cache_root /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache \
  --out_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache \
  --subsets train val test \
  --rebuild_index_only
```

Observed terminal output for the rebuild:

- `processed_sequences: 28`
- `written_patches: 18685` on an earlier rebuild pass
- later verified final full index/file count is `104928`

## Training Runs Observed

### Partial v4 training

Run name:

- `ss3dm_prior_partial_try_v4`

Artifacts:

- output dir: `outputs/ss3dm_prior/train_partial_try_v4`
- checkpoint history file: `outputs/ss3dm_prior/train_partial_try_v4/history.json`

Checkpoint summary recorded in terminal:

- `best_recon: 0.12071183541958982`
- `best_gain: 0.011888375160369006`

Selected metrics from `history.json`:

- epoch 0:
  - `train_denoise_gain_chamfer: 0.0012670921721748608`
  - `val_denoise_gain_chamfer: 0.001750400692901828`
- epoch 5:
  - `train_denoise_gain_chamfer: 0.008349073172752474`
  - `val_denoise_gain_chamfer: 0.009080756587738341`
- epoch 8:
  - `train_denoise_gain_chamfer: 0.011228661540959988`
  - `val_denoise_gain_chamfer: 0.011888375160369006`
- epoch 9:
  - `train_denoise_gain_chamfer: 0.011871280431151932`
  - `val_denoise_gain_chamfer: 0.011678406640209934`

Additional metrics present in `history.json`:

- `val_retrieval_top1` progressed from `0.6068181991577148` at epoch 0 to `0.960454523563385` at epoch 9
- `val_score_spearman` was logged each epoch and varied between negative and positive values

### Full v5 training

Run name:

- `ss3dm_prior_full_v5`

Artifacts:

- output dir: `outputs/ss3dm_prior/train_full_v5`
- wandb run directory: `outputs/ss3dm_prior/train_full_v5/wandb/run-20260410_115444-rzp4h5bw`

Observed command:

```bash
CUDA_VISIBLE_DEVICES=6 python -m ss3dm_prior.train \
  --data_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/data_default.yaml \
  --model_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/model_v5_gain.yaml \
  --train_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/train_v5_gain.yaml \
  --manifest_path /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json \
  --observed_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache \
  --town_mesh_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache \
  --patch_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache \
  --split_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/default_town_split.yaml \
  --run_name ss3dm_prior_full_v5 \
  --output_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_full_v5 \
  --wandb_project ss3dm_prior \
  --wandb_mode online
```

Terminal facts:

- The run was manually interrupted by the user with `Ctrl+C`
- Checkpoints up to `epoch_013.pt` were present at the time of inspection

Checkpoint facts from direct checkpoint inspection:

- `last.pt`
  - `epoch = 13`
  - `global_step = 73458`
  - `best_recon = 0.12096738501180403`
  - `best_gain = 0.02712792973835112`
- `best_recon.pt`
  - `best_recon = 0.12027373295541852`
  - `best_gain = 0.02712792973835112`
- `best_gain.pt`
  - `best_recon = 0.12027373295541852`
  - `best_gain = 0.027821581794736632`

Per-epoch best metric progression read from checkpoint files:

- `epoch_001.pt`: `best_gain = 0.017884891104412173`, `best_recon = 0.13021042364574298`
- `epoch_002.pt`: `best_gain = 0.020541554153388736`, `best_recon = 0.1275537605967664`
- `epoch_003.pt`: `best_gain = 0.022245349881895946`, `best_recon = 0.1258499648682592`
- `epoch_004.pt`: `best_gain = 0.022245349881895946`, `best_recon = 0.1258499648682592`
- `epoch_005.pt`: `best_gain = 0.02305178604732475`, `best_recon = 0.1250435287028304`
- `epoch_006.pt`: `best_gain = 0.024471643402921135`, `best_recon = 0.12362367134723401`
- `epoch_007.pt`: `best_gain = 0.024974602002557838`, `best_recon = 0.12312071274759731`
- `epoch_008.pt`: `best_gain = 0.025395182315585022`, `best_recon = 0.12270013243457013`
- `epoch_009.pt`: `best_gain = 0.026269136052083426`, `best_recon = 0.12182617869807172`
- `epoch_010.pt`: `best_gain = 0.026269136052083426`, `best_recon = 0.12182617869807172`
- `epoch_011.pt`: `best_gain = 0.026269136052083426`, `best_recon = 0.12182617869807172`
- `epoch_012.pt`: `best_gain = 0.02675425004628364`, `best_recon = 0.12134106470387152`
- `epoch_013.pt`: `best_gain = 0.02712792973835112`, `best_recon = 0.12096738501180403`

## Evaluation Runs Observed

### Partial v4 eval

Run name:

- `partial_try_v4_eval_best_gain`

Command used:

```bash
python -m ss3dm_prior.eval \
  --checkpoint /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_partial_try_v4/checkpoints/best_gain.pt \
  --manifest_path /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json \
  --patch_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_partial_try_v2 \
  --split_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/partial_town01_eval.yaml \
  --output_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior_eval \
  --eval_name partial_try_v4_eval_best_gain \
  --wandb_project ss3dm_prior_eval \
  --wandb_mode online
```

Summary metrics from `metrics_summary.json`:

- `patch_count: 11001`
- `sequence_count: 3`
- `town_count: 1`
- `recon_chamfer_l1: 0.12164603922448043`
- `recon_normal_cosine: 0.8328501615737982`
- `denoise_gain_chamfer: 0.011006467536724369`
- `score_mae: 0.023211722660688865`
- `score_spearman: 0.06885682118799959`
- `point_defect_mae: 0.06545837450819594`
- `retrieval_top1: 0.9044632315635681`
- `retrieval_top5: 0.965548574924469`

Per-sequence metrics from `metrics_per_sequence.csv`:

- `Town01__1000_streetsurf`
  - `patch_count: 5592`
  - `mean_denoise_gain: 0.012760756609265896`
- `Town01__150_streetsurf`
  - `patch_count: 1820`
  - `mean_denoise_gain: 0.008949667362721412`
- `Town01__300_streetsurf`
  - `patch_count: 3589`
  - `mean_denoise_gain: 0.009316133689701009`

Additional distribution facts computed from `patch_predictions.csv`:

- fraction of positive-gain patches: `0.8789200981728934`
- gain percentiles:
  - `p1: -0.005141109228134155`
  - `p5: -0.0007795542478561401`
  - `p10: -0.00010383129119873047`
  - `p25: 0.00046299397945404053`
  - `p50: 0.006587468087673187`
  - `p75: 0.018830113112926483`
  - `p90: 0.029785767197608948`
  - `p95: 0.03652337193489075`
  - `p99: 0.04894265532493591`

### Full v5 eval

Run name:

- `full_v5_eval_best_gain`

Command observed in terminal:

```bash
python -m ss3dm_prior.eval \
  --checkpoint /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/train_full_v5/checkpoints/best_gain.pt \
  --manifest_path /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json \
  --patch_cache_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache \
  --split_config /data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/default_town_split.yaml \
  --output_dir /data2/peilincai/mesh-splatting/outputs/ss3dm_prior_eval \
  --eval_name full_v5_eval_best_gain \
  --wandb_project ss3dm_prior_eval \
  --wandb_mode online
```

Summary metrics from `outputs/ss3dm_prior_eval/full_v5_eval_best_gain/metrics_summary.json`:

- `patch_count: 17640`
- `sequence_count: 4`
- `town_count: 1`
- `recon_chamfer_l1: 0.10516641875834556`
- `recon_normal_cosine: 0.863154097746365`
- `denoise_gain_chamfer: 0.03606056163674579`
- `score_mae: 0.02506895121060261`
- `score_spearman: 0.250824923681362`
- `point_defect_mae: 0.05542671634929673`
- `retrieval_top1: 0.9786847829818726`
- `retrieval_top5: 0.9949546456336975`

Per-town metrics from `metrics_per_town.csv`:

- `Town10`
  - `patch_count: 17640`
  - `sequence_count: 4`
  - `recon_chamfer_l1: 0.10516641875834556`
  - `recon_normal_cosine: 0.863154097746365`
  - `denoise_gain_chamfer: 0.03606056163674579`
  - `score_mae: 0.02506895121060261`
  - `point_defect_mae: 0.05542671634929673`

Per-sequence metrics from `metrics_per_sequence.csv`:

- `Town10__1000_streetsurf`
  - `patch_count: 6735`
  - `mean_corruption_severity: 0.34783242139979126`
  - `mean_denoise_gain: 0.043175129034869655`
  - `recon_chamfer_l1: 0.09583404321381138`
  - `recon_normal_cosine: 0.9241528289405035`
- `Town10__200_streetsurf`
  - `patch_count: 3235`
  - `mean_corruption_severity: 0.3472978503755661`
  - `mean_denoise_gain: 0.030485475033383464`
  - `recon_chamfer_l1: 0.11268471434274818`
  - `recon_normal_cosine: 0.8108677672593634`
- `Town10__300_streetsurf`
  - `patch_count: 2773`
  - `mean_corruption_severity: 0.3479567444522244`
  - `mean_denoise_gain: 0.023776476254993684`
  - `recon_chamfer_l1: 0.1229402026893041`
  - `recon_normal_cosine: 0.7372311041043891`
- `Town10__580_streetsurf`
  - `patch_count: 4897`
  - `mean_corruption_severity: 0.34704868182348236`
  - `mean_denoise_gain: 0.036914668742956035`
  - `recon_chamfer_l1: 0.10297020888216572`
  - `recon_normal_cosine: 0.8851071885983409`

Qualitative figure references recorded in `report.md`:

- `patch_panels/best_gain__Town10__200_streetsurf__tile_001469.png`
- `patch_panels/worst_gain__Town10__1000_streetsurf__tile_003245.png`
- `patch_panels/largest_score_error__Town10__580_streetsurf__tile_000030.png`
- `sequence_maps/Town10__1000_streetsurf.png`
- `retrieval_gallery/Town10__200_streetsurf__tile_002121_retrieval.png`

## Additional Operational Facts

- `train_full_v5` was started with `CUDA_VISIBLE_DEVICES=6`
- The terminal output shows the run was manually interrupted after checkpoints through `epoch_013.pt` had already been written
- `full_v5_eval_best_gain` completed successfully and wrote:
  - `metrics_summary.json`
  - `metrics_per_town.csv`
  - `metrics_per_sequence.csv`
  - `patch_predictions.csv`
  - `report.md`
  - qualitative figure directories under `patch_panels/`, `sequence_maps/`, `retrieval_gallery/`

## Test And Verification Commands Executed During Later Iterations

Observed executed checks include:

- `pytest tests/ss3dm_prior/test_model_forward.py -q`
- `pytest tests/ss3dm_prior/test_train_dataset.py -q`
- `pytest tests/ss3dm_prior/test_train_smoke.py -q`
- `pytest tests/ss3dm_prior/test_eval_smoke.py -q`
- `python -m py_compile ...`

Observed dataset verification facts:

- after the online-corruption fix:
  - repeated accesses to the same training patch yielded different corrupted samples
- after the eval/val freeze change:
  - validation/eval corruption can be held deterministic

## Current Strict-Protocol Training Configuration

Prepared but not yet documented in a completed run:

- `configs/ss3dm_prior/train_v6_full_strict.yaml`

Current facts for this config:

- uses `debug_use_all_patches_for_train_val: false`
- uses `allow_split_fallback: false`
- intended to run with the default town holdout split file

## User-Stated Target

The user-stated target to preserve in downstream analysis is:

- apply the current work to mesh
- support a more general model behavior
- example stated by the user: input a corrupted car or cat mesh/model and recover it

This target is included here as a project objective statement provided by the user. No feasibility judgment is attached in this document.
