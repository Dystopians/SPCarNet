# Stage23.6 Tuned Integrated Topology-Control Design

Date: 2026-05-02

## Goal

Run a medium-budget integrated topology-control experiment on `parking_phone_tiny` that is more meaningful than the M23.5 180-iteration trigger smoke.

The target task remains posed multi-view image and COLMAP/camera driven Mesh Splatting scene optimization, not radar-only reconstruction.

## Design Decision

M23.5 showed that the default PRISM protection policy is too conservative for short and medium training on this scene: `recent_age_iters=500` plus legacy edge/sensitivity/uncertainty and boundary risk protection can mark every triangle protected and leave no candidates.

For this tuned run:

- keep online W&B;
- keep PRISM candidate pruning inside training;
- keep counterfactual gate enabled;
- keep geometry/render/orientation keep protections active at a higher threshold than default;
- disable the legacy all-protecting path by setting recent age to `100`, edge/sensitivity/uncertainty protected thresholds to `1.1`, and boundary/nonmanifold risk values to `0.0`;
- use small candidate prune rounds instead of large post-hoc pruning;
- keep final cleanup disabled, so any topology change is attributable to scheduled PRISM rounds.

This is intentionally a medium diagnostic row, not a final paper-budget result.

## Run Configuration

- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage23_6_tuned_integrated_topology/tuned_medium_2000iter/`
- iterations: `2000`
- GPU: `1`
- W&B project: `spcarnet_meshprior`
- W&B group: `parking_stage23_6_tuned_integrated_topology`
- W&B name: `tuned_medium_2000iter`
- PRISM schedule:
  - geometry acquisition until iteration `300`
  - stats collection `250` iterations
  - dead rounds `0`
  - candidate rounds `2`
  - prune ratio per candidate round `0.01`
  - recovery after commit `250`
  - post-commit recollect `120`
  - final finetune `500`

## Evaluation

After training:

1. Render test views with `render.py`.
2. Compute independent `metrics.py` PSNR/SSIM/LPIPS.
3. Compute COLMAP proxy geometry with `evaluate_geometry_colmap.py`.
4. Collect PRISM round metadata and final-cleanup state with the Stage23.5 collector.
5. Compare against the M23.5 mechanism smoke, M21.5 topology-controlled row, and clean/current baselines only with topology counts visible.

## Gate

`PASS` if the run completes with online W&B, PRISM round metadata exists, at least one topology edit commits or rejects through the gate with explicit metadata, render/geometry/final-cleanup artifacts exist, and collector/reporting is complete.

`SOFT PASS` if the run completes and all artifacts exist, but topology edits remain too conservative or quality is not competitive.

`FAIL` if training crashes, W&B is missing without fallback, or rollback/final-cleanup accounting is absent.
