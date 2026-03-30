# PRISM-Prune Recipe (Parking-Ground)

This note gives a minimal and reproducible way to run PRISM ablations and compare:

1. geometry-first without PRISM  
2. geometry-first + dead prune only  
3. geometry-first + full PRISM  
4. geometry-first + full PRISM + optional ground/ROI protection

## Core Idea

PRISM-Prune is a multi-evidence, rollback-safe pruning pipeline:

- collect per-triangle statistics
- score triangles by utility/risk/redundancy
- run conservative dead prune and candidate prune rounds
- gate candidate prune with counterfactual simulation
- after each prune-recovery round, run global validation gate
- rollback the whole round if validation regresses beyond thresholds

## Score Definitions

Default score composition:

- `utility_t = 0.30 * vis_t + 0.25 * sens_t + 0.20 * geo_t + 0.15 * viewdiv_t + 0.10 * edge_t`
- `risk_t = max(unc_t, recent_t, boundary_t, nonmanifold_t, optional_groundprotect_t)`
- `redund_t = 0.70 * flat_t + 0.30 * coplanar_t`
- `prune_score_t = redund_t * (1 - utility_t) * (1 - risk_t)`

All weights/thresholds are configurable by `--prism_*` arguments.

## Triangle States

- `PROTECTED`: high-importance or high-risk triangles, excluded from candidate ranking.
- `DEAD`: near-zero contribution triangles, can be pruned in conservative fast path.
- `SUSPICIOUS`: conflicting evidence triangles; monitored, not first-choice prune target.
- `CANDIDATE`: active, non-protected triangles eligible for candidate ranking.

## Counterfactual Gate Defaults

Candidate prune is accepted only if calibration-set degradation stays within:

- `delta_psnr >= -0.05 dB`
- `delta_mae <= +0.002`
- `delta_absrel <= +0.0008`
- `delta_mean_angle <= +0.3 deg`
- `changed_pixel_ratio <= 0.005`

## Global Validation / Rollback Defaults

At each prune-recovery round end, compare against current stage-best:

- geometry regression:
  - `AbsRel` relative degradation `> 1%`, or
  - `mean_angle` degradation `> 0.4 deg`
- visual regression:
  - `PSNR` drop `> 0.10 dB`, or
  - `MAE` increase `> 0.003`

Any trigger causes full round rollback (round-pre snapshot restore).

## Training Stages

PRISM stages are:

1. geometry acquisition
2. stats collection (optional topology freeze)
3. dead prune round(s)
4. candidate prune round(s), counterfactual-gated
5. recovery fine-tune after each committed prune
6. final fine-tune

## Fair Benchmark Protocol

Use the same:

- scene path
- split file
- training iterations
- rendering/eval commands
- evaluation split (`--split_strategy file --split_file ...`)

Keep test set untouched for PRISM dev-validation when split file has dropped buffer views.

## Run Commands

Set shared env once:

```bash
export SCENE_PATH=/abs/path/to/parking_scene
export SPLIT_FILE=/abs/path/to/parking_scene/sparse/0/split_outoftrain_v1.json
export MODEL_ROOT=/data2/peilincai/mesh-splatting/models
export RUN_TAG=parking_phone_tiny
export ITERATIONS=30000
```

Run four settings:

```bash
bash scripts/parking_ground/run_geom_first_no_prism.sh
bash scripts/parking_ground/run_geom_first_dead_only.sh
bash scripts/parking_ground/run_geom_first_full_prism.sh
bash scripts/parking_ground/run_geom_first_full_prism_ground_protect.sh
```

Run full end-to-end suite (training + fair benchmark + qualitative panels):

```bash
bash scripts/parking_ground/run_full_practice_suite.sh
```

Run full suite in parallel on 4 auto-selected idle GPUs:

```bash
bash scripts/parking_ground/run_full_practice_suite_auto_gpu.sh
```

This suite additionally includes the previous grounding method run:

- `scripts/parking_ground/run_geom_first_grounding.sh`

Run matched benchmark (reuses `render.py`, `metrics.py`, `evaluate_geometry_colmap.py`):

```bash
python scripts/parking_ground/benchmark_prism_runs.py \
  --repo_root . \
  --scene_path "$SCENE_PATH" \
  --split_file "$SPLIT_FILE" \
  --run no_prism="$MODEL_ROOT/${RUN_TAG}_geom_first_no_prism" \
  --run dead_only="$MODEL_ROOT/${RUN_TAG}_geom_first_dead_only" \
  --run full_prism="$MODEL_ROOT/${RUN_TAG}_geom_first_full_prism" \
  --run full_prism_ground="$MODEL_ROOT/${RUN_TAG}_geom_first_full_prism_ground_protect"
```

## Outputs and Where to Inspect

Per benchmark run:

- machine-readable: `benchmarks/prism_parking_ground/<timestamp>/benchmark_results.json`
- human-readable: `benchmarks/prism_parking_ground/<timestamp>/benchmark_summary.md`

Per training run debug artifacts:

- `.../prism_round_checkpoints/` (round pre/post checkpoints)
- `.../prism_validation/validation_iter_*.json|.md` (validation + rollback traces)
- `.../prism_debug/` (score/stat debug dumps)
- `.../geometry_eval_colmap/` (geometry eval json)

## WandB Real-Time Supervision (non-duplicate)

Training now logs filtered scalar streams (unchanged values are skipped), including:

- train/core: `train_loss/*`, `train/*`
- mesh/topology: `mesh/*`
- loss decomposition: `loss_components/*`
- PRISM: `prism/*`, `prism/val_*`, `prism/val_delta_*`
- ground: `ground_reg/*`, `ground_assoc/*`
- fixed qualitative views: `fixed_eval/*`

Useful controls:

- `--enable_wandb`
- `--wandb_scalar_log_interval 10`
- `--wandb_image_log_interval 1000`
- `--wandb_disable_fixed_views` (max speed mode; disables periodic heavy qual/eval block)

## Ground/ROI Compatibility

Ground/ROI is optional:

- PRISM main gate uses global metrics and does not depend on ground module.
- If `ground_mask` exists, ROI-only validation metrics are emitted as analysis-only breakdown.
- Optional protection switches:
  - `--prism_use_ground_protect`
  - `--prism_use_roi_protect`
