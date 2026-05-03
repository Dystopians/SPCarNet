# MeshSplatOpt R21.01-R21.03 Residual Patch Snap Report

Date: 2026-05-03

## Objective

R20 showed that isolated residual snap vertices can improve depth under medium recovery, but they are too sparse and can lose appearance quality. R21 introduces a patch-level residual repair primitive that expands seed residual snap vertices to neighboring mesh vertices while remaining a checkpoint-compatible `SNAP_VERTICES` edit.

## Implementation

Added:

`scripts/car_model/meshsplatopt_expand_snap_edit_to_patch.py`

The script:

- loads a seed `SNAP_VERTICES` edit;
- builds mesh vertex adjacency from checkpoint triangle indices;
- expands each seed to a local k-hop/radius patch;
- applies distance-weighted fractions of the seed displacement to neighboring vertices;
- emits a single checkpoint-compatible patch `SNAP_VERTICES` edit with an audit report.

This keeps rollback and checkpoint-gate compatibility because no topology format changes are required.

## Patch Candidate

Seed edit:

`outputs/carnet/meshsplatopt/stageR18_02_parking_train_residual_snap_selection/selected_residual_snap_edit.json`

Patch output:

`outputs/carnet/meshsplatopt/stageR21_02_parking_residual_patch_snap_r15_selection`

Selection:

- seed vertices: `16`
- patch vertices: `95`
- affected faces: `217`
- neighbor hops: `1`
- max radius: `15.0`
- falloff radius: `7.5`
- max displacement: `0.074180`
- mean displacement: `0.018138`

## Held-Out Gate

Gate output:

`outputs/carnet/meshsplatopt/stageR21_02_parking_residual_patch_snap_r15_selection/gate/render_backed_checkpoint_gate_report.json`

Result: `PASS`

| metric | delta |
| --- | ---: |
| triangles | `0` |
| vertices | `0` |
| PSNR | `+0.00000095` |
| SSIM | `+0.00000009` |
| LPIPS | `-0.00000089` |
| AbsRel | `0.0` |
| Depth MAE | `0.0` |
| Normal mean deg | `-0.00000130` |

## W&B Recovery

Recovery output:

`outputs/carnet/meshsplatopt/stageR21_03_parking_residual_patch_snap_freeze_skip_delaunay_2000to2200`

W&B:

`https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/76fgy4z5`

Protocol:

- load iteration: `2000`
- train until: `2200`
- densify until: `2000`
- restricted Delaunay: skipped
- GPU: `4`
- train/render/metrics exit codes: `0/0/0`

## Equal-Budget Result

| run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal mean deg ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline continuation | `12.331465` | `0.298222` | `0.622323` | `0.409263` | `4.300273` | `52.595639` |
| single residual snap | `12.342549` | `0.298893` | `0.622299` | `0.408892` | `4.302941` | `52.354489` |
| patch residual snap | `12.329646` | `0.298382` | `0.622157` | `0.409988` | `4.303037` | `52.586082` |

Patch residual snap minus baseline:

- PSNR: `-0.001819`
- SSIM: `+0.000161`
- LPIPS: `-0.000166`
- AbsRel: `+0.000724`
- Depth MAE: `+0.002764`
- Normal mean deg: `-0.009557`

## Decision

`PATCH_SNAP_GATE_PASS_RECOVERY_MIXED`

Patch expansion fixes an architectural weakness: the system now has a checkpoint-compatible patch-level repair primitive that passes held-out render and geometry gate. However, this first patch policy does not yet dominate either baseline continuation or single residual snap after recovery. It improves LPIPS and normal angle over baseline, but loses PSNR and depth metrics. The next repair step should use residual-cluster optimization with a render/depth objective instead of simply diffusing seed displacement to neighbors.
