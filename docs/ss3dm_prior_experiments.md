# SS3DM Prior Experiments

## Step 6 Debug Training

Minimal debug launch:

```bash
bash scripts/ss3dm_prior/train_debug.sh
```

Direct CLI form:

```bash
python -m ss3dm_prior.train \
  --data_config configs/ss3dm_prior/data_default.yaml \
  --model_config configs/ss3dm_prior/model_default.yaml \
  --train_config <debug_train_config.yaml> \
  --manifest_path outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json \
  --observed_cache_dir outputs/ss3dm_prior/observed_cache_debug \
  --town_mesh_cache_dir outputs/ss3dm_prior/town_mesh_cache_smoke \
  --patch_cache_dir outputs/ss3dm_prior/teacher_patch_cache_debug_val \
  --split_config configs/ss3dm_prior/splits/debug_town_split.yaml \
  --run_name ss3dm_prior_debug \
  --output_dir outputs/ss3dm_prior/train_debug_real \
  --wandb_project ss3dm_prior \
  --wandb_mode offline
```

## What To Inspect In wandb

Key scalar charts:

- `train/total_loss`
- `train/recon_chamfer_loss`
- `train/point_defect_loss`
- `epoch/val_recon_chamfer_l1`
- `epoch/val_denoise_gain_chamfer`
- `epoch/val_score_mae`
- `epoch/val_score_spearman`
- `epoch/val_retrieval_top1`
- `epoch/val_retrieval_top5`

Key qualitative images:

- `viz/patch_denoise_panel/*`
- `viz/sequence_improvement_map`
- `viz/retrieval_gallery`

Interpretation:

- `patch_denoise_panel` should show corrupted input moving toward the clean target, with defect heatmap focusing on missing or damaged regions.
- `sequence_improvement_map` should reveal whether predicted difficulty aligns with actual denoise gain across one validation sequence.
- `retrieval_gallery` should show whether corrupted-patch embeddings retrieve the correct clean local geometry neighborhood.

## Checkpoint Policy

Saved checkpoints:

- `last.pt`
- `best_recon.pt`
- `best_gain.pt`

Selection rules:

- `best_recon.pt`: best validation `recon_chamfer_l1` (lower is better)
- `best_gain.pt`: best validation `denoise_gain_chamfer` (higher is better)

## Current Debug Run Snapshot

Real one-epoch debug run output:

- output dir: `outputs/ss3dm_prior/train_debug_real`
- local offline wandb run: `outputs/ss3dm_prior/train_debug_real/wandb/...`
- local patch panel example:
  - `outputs/ss3dm_prior/train_debug_real/visualizations/epoch_000/Town02__150_streetsurf__tile_000002_panel.png`
- local sequence map example:
  - `outputs/ss3dm_prior/train_debug_real/visualizations/epoch_000/Town02__150_streetsurf_sequence_map.png`

Observed best-checkpoint summary from the debug run:

- `best_recon: 1.0407075881958008`
- `best_gain: -0.9008909314870834`

## Step 7 Debug Eval

Minimal eval launch:

```bash
bash scripts/ss3dm_prior/eval_default.sh
```

Direct CLI form:

```bash
python -m ss3dm_prior.eval \
  --checkpoint outputs/ss3dm_prior/train_debug_real/checkpoints/best_recon.pt \
  --manifest_path outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json \
  --patch_cache_dir outputs/ss3dm_prior/teacher_patch_cache_debug_val \
  --split_config outputs/ss3dm_prior/train_debug_real/eval_debug_town02.yaml \
  --output_dir outputs/ss3dm_prior_eval \
  --eval_name smoke_debug_eval \
  --wandb_project ss3dm_prior_eval \
  --wandb_mode disabled
```

## Eval Output Structure

Expected files under `outputs/ss3dm_prior_eval/<eval_name>/`:

- `metrics_summary.json`
- `metrics_per_town.csv`
- `metrics_per_sequence.csv`
- `patch_predictions.csv`
- `patch_panels/*.png`
- `sequence_maps/*.png`
- `retrieval_gallery/*.png`
- `report.md`

## What To Inspect In Eval Outputs

Key numeric artifacts:

- `metrics_summary.json` for global checkpoint comparison
- `metrics_per_town.csv` for town-level generalization gaps
- `metrics_per_sequence.csv` for sequence-level variation
- `patch_predictions.csv` for patch-level debugging and ranking

Key qualitative artifacts:

- `patch_panels/best_gain__*.png`
- `patch_panels/worst_gain__*.png`
- `patch_panels/largest_score_error__*.png`
- `sequence_maps/*.png`
- `retrieval_gallery/*_retrieval.png`

Interpretation:

- `best_gain` shows where reconstruction improved the most relative to the corrupted input.
- `worst_gain` highlights failure cases where denoising hurt or failed to help.
- `largest_score_error` shows whether the difficulty head is missing obvious hard or easy cases.
- `sequence_maps` make it easier to see whether predicted difficulty aligns with actual denoise gain spatially.
- `retrieval_gallery` shows whether the learned embedding retrieves the correct clean local neighborhood.

## Current Debug Eval Snapshot

Real smoke eval output:

- output dir: `outputs/ss3dm_prior_eval/smoke_debug_eval`
- report: `outputs/ss3dm_prior_eval/smoke_debug_eval/report.md`
- patch panel example:
  - `outputs/ss3dm_prior_eval/smoke_debug_eval/patch_panels/best_gain__Town02__150_streetsurf__tile_000001.png`
- sequence map example:
  - `outputs/ss3dm_prior_eval/smoke_debug_eval/sequence_maps/Town02__150_streetsurf.png`
- retrieval example:
  - `outputs/ss3dm_prior_eval/smoke_debug_eval/retrieval_gallery/Town02__150_streetsurf__tile_000004_retrieval.png`

Observed summary metrics from the smoke eval:

- `recon_chamfer_l1: 1.0020790174603462`
- `recon_normal_cosine: 0.2569358628243208`
- `denoise_gain_chamfer: -0.8718227623030543`
- `score_mae: 0.15290501154959202`
- `score_spearman: 0.07142857142857144`
- `point_defect_mae: 0.09508668165653944`
- `retrieval_top1: 0.125`
- `retrieval_top5: 0.625`
