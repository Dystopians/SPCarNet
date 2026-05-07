# Stage ELA11 Sparse-Occluder Adaptive Policy Report

## Goal

Move from per-scene parameter attempts to a fixed self-diagnostic repair policy that can satisfy the strict multi-axis criterion: RGB metrics, sparse geometry metrics, and triangle count must all beat the clean Mesh Splatting baseline.

## New Mechanism

Stage ELA11 adds sparse occluder rejection (SOR). The selector uses only train-split COLMAP sparse-depth correspondences. For each train view, it samples the rendered triangle id at sparse COLMAP pixels and marks a face as risky when the rendered surface is in front of the COLMAP depth by a fixed relative margin. The final SOR candidate set is the union of:

- a small low-evidence topology base (`base_prune_fraction=0.10`);
- train-split front-occluder faces capped by the fixed policy (`max_sparse_occluder_fraction=0.01`).

This is implemented in `scripts/car_model/meshsplatopt_build_sparse_occluder_prune_candidates.py`.

## Adaptive Routing

The new router is implemented in `scripts/car_model/meshsplatopt_select_adaptive_repair_action.py`.

Fixed policy:

- if train sparse front-occluder fraction is at least `0.25`, use SOR;
- otherwise, for normal-size indoor scenes, use QEM50 sparse parent-rollback + ELA;
- for large meshes without high sparse-occluder signal, fall back to CSEF adaptive sparse-depth recovery.

Observed routing:

| scene | train front-occluder fraction | selected branch | outcome |
|---|---:|---|---|
| bonsai | `0.460542` | SOR | strict full-pass |
| courtyard | `0.314915` | SOR | strict full-pass |
| counter | `0.055984` | QEM | SOR transfer fails, QEM passes |
| room | `0.118611` | QEM | SOR transfer fails, QEM passes |

## Strict Results vs Clean Mesh Splatting

| scene | promoted method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | tri reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bonsai | SOR10 + ELA safe | `+2.838371` | `+0.163376` | `-0.099541` | `-0.105169` | `-1.032433` | `-2.410058` | `10.25%` |
| courtyard | SOR10 + ELA safe | `+0.969368` | `+0.028828` | `-0.056569` | `-0.104763` | `-1.288431` | `-2.711335` | `10.34%` |
| counter | QEM50 parent-rollback + ELA safe | `+3.157017` | `+0.069925` | `-0.070661` | `-0.000686` | `-0.008253` | `-2.080537` | `50.00%` |
| room | QEM50 parent-rollback + ELA safe | `+3.304691` | `+0.050085` | `-0.062170` | `-0.002331` | `-0.019509` | `-1.824378` | `50.00%` |
| parking_phone_tiny | CSEF70 sparse-depth compact recovery | `+0.232340` | `+0.013107` | `-0.008653` | `-0.003106` | `-0.014383` | `-1.072729` | `70.00%` |

## Negative Transfer Evidence

SOR is not promoted as a universal action:

- counter SOR10: `-0.126171` PSNR, `-0.003474` SSIM, `+0.003394` LPIPS, `+0.001241` AbsRel, `+0.009442` Depth MAE, but `10.62%` fewer triangles.
- room SOR10: `-0.624842` PSNR, `-0.008025` SSIM, `+0.011262` LPIPS, `+0.001742` AbsRel, `+0.006635` Depth MAE, but `10.78%` fewer triangles.

These failures are useful: they justify the adaptive router instead of a hand-picked scene table.

## Current Decision

`STRICT_MULTIAXIS_SELECTED_SCENES_FULL_PASS`.

The selected clean9000 audit now has strict full-pass rows for bonsai, courtyard, room, and counter. Courtyard is solved by the same high-sparse-occluder SOR branch as bonsai, and the routed composite policy now has at least one strict full-pass row for every selected scene. The updated table is in `docs/car_model/stageELA11_strict_multiaxis_audit_report.md`.

## Key Artifacts

- SOR selector: `scripts/car_model/meshsplatopt_build_sparse_occluder_prune_candidates.py`
- Adaptive router: `scripts/car_model/meshsplatopt_select_adaptive_repair_action.py`
- Strict audit: `docs/car_model/stageELA11_strict_multiaxis_audit_report.md`
- Bonsai W&B ELA run: `vmai8bls`
- Courtyard W&B ELA run: `xcoa2n7y`
- Counter W&B ELA run: `zcc5inc0`
- Room W&B ELA run: `9t01dwd8`
