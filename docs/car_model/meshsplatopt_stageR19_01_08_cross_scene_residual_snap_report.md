# MeshSplatOpt R19.01-R19.08 Cross-Scene Residual Snap Report

Date: 2026-05-03

## Objective

R18 established that train-residual local snap can beat same-budget continuation on parking for most metrics. R19 tests whether the selector generalizes beyond parking, fixes the manual camera-offset weakness, calibrates the proposal uncertainty threshold, and runs same-source 200-step W&B recovery on courtyard and bonsai.

## Implementation Updates

Updated `scripts/car_model/meshsplatopt_select_checkpoint_residual_snap_edit.py`:

- camera index offset is now inferred automatically from `cameras.json` and the number of rendered train/test residual views;
- reports now include `camera_index_offset_mode`, `num_render_views`, `accepted_before_risk_filter_count`, `rejection_reasons`, and `risk_filter_rejections`;
- default `--max_proposal_uncertainty` is calibrated from `0.35` to `0.55`.

The auto-offset check reproduced the parking offset:

- parking train render views: `371`
- cameras: `425`
- inferred offset: `54`

## Cross-Scene Selection

Strict `0.35` uncertainty was too conservative outside parking:

| scene | model | inferred offset | train views | strict status | strict selected |
| --- | --- | ---: | ---: | --- | ---: |
| courtyard | Stage35 retained refresh | `5` | `33` | `NO_CANDIDATE` | `0` |
| bonsai | Stage35 retained refresh | `37` | `255` | `NO_CANDIDATE` | `0` |

Relaxed/calibrated `0.55` uncertainty selected 16 vertices on both scenes:

| scene | status | candidate faces | candidate vertices | scored vertices | valid proposals | selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| courtyard | `PASS` | `2548` | `4996` | `4188` | `3069` | `16` |
| bonsai | `PASS` | `15833` | `7101` | `3676` | `3037` | `16` |

## Held-Out Gate

Both calibrated cross-scene edits passed the render-backed checkpoint gate before training.

| scene | status | PSNR delta | SSIM delta | LPIPS delta | AbsRel delta | Depth MAE delta | Normal delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| courtyard | `PASS` | `-0.000409` | `-0.000013` | `+0.000014` | `0.0` | `0.0` | `-0.000597` |
| bonsai | `PASS` | `-0.000010` | `-0.000002` | `+0.000003` | `0.0` | `0.0` | `+0.0000003` |

Topology stayed unchanged in both scenes:

- courtyard: `101913` triangles, `168817` vertices
- bonsai: `633275` triangles, `1047525` vertices

## W&B Recovery

All recovery runs used GPU 4, W&B online logging, load iteration `2000`, train to `2200`, densify until `2000`, and `--skip_restricted_delaunay`.

| scene | run | W&B |
| --- | --- | --- |
| courtyard | baseline continuation | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ajvqp7ou` |
| courtyard | residual snap | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mhjbnm2t` |
| bonsai | baseline continuation | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/b9miy649` |
| bonsai | residual snap | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/p33pm98r` |

## Equal-Budget Results

### Courtyard

| run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal mean deg ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline continuation | `15.993344` | `0.527149` | `0.566524` | `0.145460` | `1.749319` | `32.482345` |
| residual snap | `15.991000` | `0.527132` | `0.566341` | `0.145153` | `1.747247` | `32.768190` |

Residual snap minus baseline:

- PSNR: `-0.002344`
- SSIM: `-0.000018`
- LPIPS: `-0.000183`
- AbsRel: `-0.000306`
- Depth MAE: `-0.002072`
- Normal mean deg: `+0.285845`

### Bonsai

| run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal mean deg ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline continuation | `13.475551` | `0.316892` | `0.588726` | `0.444263` | `4.530295` | `48.266313` |
| residual snap | `13.475065` | `0.316831` | `0.588614` | `0.444109` | `4.528911` | `48.230867` |

Residual snap minus baseline:

- PSNR: `-0.000485`
- SSIM: `-0.000061`
- LPIPS: `-0.000112`
- AbsRel: `-0.000154`
- Depth MAE: `-0.001383`
- Normal mean deg: `-0.035446`

## Decision

`CROSS_SCENE_RESIDUAL_SNAP_GATE_PASS_RECOVERY_MIXED_POSITIVE`

R19 is a meaningful generalization step:

- residual snap now works on parking, courtyard, and bonsai;
- auto camera offset removes a manual cross-scene failure mode;
- `0.55` uncertainty is empirically gate-safe on courtyard and bonsai;
- both cross-scene recovery runs improve LPIPS, AbsRel, and Depth MAE, while render PSNR/SSIM move by tiny negative amounts;
- bonsai also improves normal mean angle, while courtyard normal mean worsens despite better depth.

This is not yet a strong headline result. The effect sizes remain small, and the local snap edit is still too sparse. The next necessary method step is patch-level residual repair with clustered vertices or fill/split proposals, using the same CSEF gate rather than isolated vertex snaps.
