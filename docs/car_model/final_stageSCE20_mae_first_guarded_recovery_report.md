# Stage SCE20 MAE-First Guarded Recovery Report

Date: 2026-05-06

Decision: `SCE20_NEGATIVE_BUT_POLICY_GUARD_CAUGHT`

## Goal

Try one more controlled courtyard recovery from the current SCE7 best checkpoint, focused on the remaining Depth MAE gap. The experiment used lower vertex LR, MAE-space parent rollback, regressed-only sentinel sampling, cluster balancing, and stronger normal anchoring.

## Run

Corrected run:

`outputs/carnet/meshsplatopt/final_stageSCE20_mae_first_guarded_recovery/courtyard/mae_rollback_low_lr_28600to28720_seed0_v2/recovery_model`

- W&B: `g500vmma`
- source path: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`
- start: SCE7 best iteration 28600
- final: 28720
- topology unchanged: `true`
- sparse lambda: `0.001`
- rollback loss: `mae`
- rollback lambda: `0.75`
- rollback cache: current-residual train sentinel cache
- rollback top-k clusters: `12`
- LR triangles points init: `0.001`
- render normal anchor: `0.02`

One failed launch happened first because the source path was incorrectly set to `mipnerf360/courtyard`; that W&B run is `a9lt2r49` and produced no useful training result.

## Result

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
|---|---:|---:|---:|---:|---:|---:|
| F82 26000 | 12.198611 | 0.308649 | 0.566687 | 0.301884 | 3.339872 | 40.215702 |
| SCE7 28600 | 12.610288 | 0.338174 | 0.560068 | 0.298901 | 3.341660 | 39.368305 |
| SCE20 28720 | 12.615541 | 0.338794 | 0.559708 | 0.298987 | 3.342733 | 39.361471 |

Delta SCE20 minus SCE7:

- PSNR: `+0.005254`
- SSIM: `+0.000620`
- LPIPS: `-0.000360`
- AbsRel: `+0.000086`
- Depth MAE: `+0.001074`
- Normal: `-0.006834`

The run improves render and normal slightly, but worsens both sparse depth metrics. It is therefore rejected.

## Policy Guard

SCE20 extends the guarded policy with a full parent-Pareto acceptance guard. The SCE20 candidate is rejected by:

`outputs/carnet/meshsplatopt/final_stageSCE20_mae_first_guarded_recovery/courtyard/mae_rollback_low_lr_28600to28720_seed0_v2/policy_guard/contract/sce_policy_decision.json`

Decision:

- action: `accept_parent_noop`
- execute recovery: `false`
- reason: `parent_pareto_guard_failed`
- failed metrics: `absrel_above_parent`, `depth_mae_above_parent`

The guarded two-row table is:

`outputs/carnet/meshsplatopt/final_stageSCE20_mae_first_guarded_recovery/guarded_policy_table_courtyard_bonsai/stageSCE20_guarded_policy_report.md`

It shows courtyard and bonsai both become safe no-op rows under guarded policy, with no strict improvement claim.

## Interpretation

This stage did not solve the remaining courtyard depth gap. It did, however, close an important reliability hole: a candidate that improves RGB but worsens sparse geometry is now automatically rejected by policy, instead of relying on manual reviewer-style judgment after the fact.

The remaining bottleneck is still the same: held-out sparse depth on courtyard has a small but stubborn local regression that current rollback losses do not fix without harming another geometry metric.
