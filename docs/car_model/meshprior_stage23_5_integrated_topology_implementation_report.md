# Stage23.5 Integrated Topology-Control Implementation Report

Date: 2026-05-02

Gate: `PASS`

## Clarified Task

The target is not radar-only mesh reconstruction. The current defensible task setting is:

> posed multi-view images plus COLMAP/camera geometry plus Mesh Splatting scene mesh optimization, with object/scene priors used as proposal and topology-control signals.

This is a reasonable NeurIPS-facing assumption because Mesh Splatting already depends on calibrated image observations, and COLMAP sparse geometry is a standard scene-evidence source. Radar or scans can be positioned later as optional geometric evidence, but the current codebase should not claim a radar-only pipeline.

## What Was Implemented

- Added `scripts/car_model/meshprior_run_stage23_5_integrated_topology_smoke.sh`.
- Added `scripts/car_model/meshprior_collect_stage23_5_integrated_topology.py`.
- Added `scripts/car_model/smoke_test_meshprior_stage23_5_integrated_topology_collector.py`.
- Added `docs/car_model/meshprior_stage23_5_integrated_topology_design.md`.
- Reused existing training-time PRISM scheduling, candidate pruning, counterfactual gate, rollback metadata, final-cleanup summary, render metrics, and COLMAP proxy geometry evaluation.

## Experiments

### Debug Run A: Default-Protected PRISM Smoke

- name: `prism_smoke_800iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/5ekk5gjz`
- result: PRISM scheduling and evaluation artifacts were produced, but no topology edit committed.
- cause: all triangles were protected in the short early smoke; the candidate set was empty.
- independent render metrics: PSNR `12.053230`, SSIM `0.340518`, LPIPS `0.623179`.
- COLMAP proxy depth AbsRel: `0.301412`.
- topology: `64497` triangles, `193491` vertices.

### Debug Run B: Partially Relaxed Candidate Smoke

- name: `prism_candidate_smoke_350iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/esyvtvwn`
- result: still no topology edit committed.
- cause: legacy edge/sensitivity/uncertainty/risk protection thresholds still protected all candidates.
- note: this run is a debugging artifact only; post-eval was superseded by the successful direct run below.

### Successful Trigger Run

- name: `prism_unprotected_trigger_180iter`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/an7l2ec0`
- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage23_5_integrated_topology/prism_unprotected_trigger_180iter/`
- training-time topology edit:
  - PRISM round iteration: `141`
  - mode: `candidate`
  - committed: `true`
  - rollback: `0`
  - pre-prune triangles: `64497`
  - post-prune triangles: `63208`
  - removed triangles: `1289`
- independent render metrics at iteration `180`:
  - PSNR: `10.790648`
  - SSIM: `0.284250`
  - LPIPS: `0.645548`
- COLMAP proxy geometry at iteration `180`:
  - depth AbsRel: `0.327274`
  - depth MAE: `3.666642`
  - normal mean angle: `51.771524`
- final checkpoint topology:
  - triangles: `63208`
  - vertices: `193491`
- final cleanup:
  - enabled: `false`
  - cleanup executed: `false`
  - pre/post cleanup triangles: `63208 -> 63208`

Collector output:

```text
gate: PASS
rounds: 1
committed: 1
```

## Commands

The successful trigger run used the current training script with training-time W&B:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py \
  -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view \
  -m outputs/carnet/meshprior/parking_phone_tiny/stage23_5_integrated_topology/prism_unprotected_trigger_180iter/model \
  --images images --eval --iterations 180 --test_iterations 180 --save_iterations 180 \
  --checkpoint_iterations 180 --resolution 4 --enable_wandb \
  --wandb_project spcarnet_meshprior --wandb_group parking_stage23_5_integrated_topology \
  --wandb_name prism_unprotected_trigger_180iter \
  --enable_prism_pruning --prism_collect_stats --prism_use_counterfactual_gate \
  --prism_candidate_rounds 1 --prism_candidate_prune_ratio_per_round 0.02 \
  --prism_recent_age_iters 0 --prism_thresh_protected_edge 1.1 \
  --prism_thresh_protected_geo 1.1 --prism_thresh_protected_sens 1.1 \
  --prism_thresh_protected_unc 1.1 --prism_boundary_risk_value 0.0 \
  --prism_nonmanifold_risk_value 0.0 --prism_protected_dilation_rings 0 \
  --prism_disable_final_cleanup_prune --enable_sparse_colmap_depth_loss
```

Independent post-eval:

```bash
CUDA_VISIBLE_DEVICES=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view \
  -m outputs/carnet/meshprior/parking_phone_tiny/stage23_5_integrated_topology/prism_unprotected_trigger_180iter/model \
  --images images --eval --iteration 180 --skip_train --quiet

CUDA_VISIBLE_DEVICES=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py \
  -m outputs/carnet/meshprior/parking_phone_tiny/stage23_5_integrated_topology/prism_unprotected_trigger_180iter/model

CUDA_VISIBLE_DEVICES=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py \
  -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view \
  -m outputs/carnet/meshprior/parking_phone_tiny/stage23_5_integrated_topology/prism_unprotected_trigger_180iter/model \
  --images images --eval --iteration 180 --max_points_per_view 500 \
  --output outputs/carnet/meshprior/parking_phone_tiny/stage23_5_integrated_topology/prism_unprotected_trigger_180iter/model/geometry_eval_colmap/iter_180.json
```

## Interpretation

This stage proves that topology control can be integrated into the optimization loop and can commit a PRISM candidate prune with metadata and rollback accounting. It does not prove quality improvement: `180` iterations is a mechanism smoke, and the successful trigger deliberately relaxed protection thresholds to verify the commit path.

The next useful task is a tuned medium integrated-topology run that restores conservative protection/gate thresholds enough for paper relevance while still allowing scheduled candidate edits.

## Verification

- collector gate: `PASS`
- `scripts/car_model/smoke_test_meshprior_stage23_5_integrated_topology_collector.py`: PASS
- `python -m compileall scripts/car_model ss3dm_prior -q`: PASS
- `git diff --check`: PASS
