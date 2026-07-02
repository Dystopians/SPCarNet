# v345e Source-Confirmed PSNR-Dominant Pairwise Certificate

Date: 2026-07-02

## Summary

v345e adds a bounded source-heldout pairwise certificate for cases where PSNR
evidence is strong but SSIM evidence is only slightly negative. The final
submitted policy is not a parameter scan: target-time SSIM tolerance is allowed
only when the source-heldout pairwise policy itself needed and passed the same
PSNR-dominant SSIM tolerance. This source-confirmed constraint keeps the useful
treehill repair while preventing the v345d bicycle false promotion.

Implementation:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New interface:

```text
--pairwise_dominance_enable_psnr_dominant_ssim_tolerance
--pairwise_dominance_psnr_dominant_local_ssim_floor
--pairwise_dominance_psnr_dominant_source_ssim_floor
--pairwise_dominance_psnr_dominant_min_predicted_psnr_delta
--pairwise_dominance_psnr_dominant_min_predicted_ssim_delta
--pairwise_dominance_psnr_dominant_min_local_psnr_delta
--pairwise_dominance_psnr_dominant_min_local_cvar_delta
--pairwise_dominance_psnr_dominant_min_local_min_delta
--pairwise_dominance_psnr_dominant_min_positive_fraction
--pairwise_dominance_psnr_dominant_min_source_psnr_delta
--pairwise_dominance_psnr_dominant_min_source_cvar_delta
--pairwise_dominance_psnr_dominant_min_source_min_delta
--pairwise_dominance_psnr_dominant_min_source_accept_fraction
```

The report now records both source-level and per-target usage:

```text
pairwise_dominance_policy.source_psnr_dominant_ssim_tolerance_used
per_view[].pairwise_dominance_diagnostics.best_diagnostics.psnr_dominant_ssim_tolerance_used
per_view[].pairwise_dominance_diagnostics.best_diagnostics.source_psnr_dominant_ssim_tolerance_used
```

## Why v345d Was Not Enough

The first clean probe, v345d, used the PSNR-dominant tolerance without the
global OOD/max-blend guards that had caused the v345c room regression. It fixed
treehill and preserved room/bonsai, but full focus6 exposed one small bicycle
regression:

```text
v345d-v343e bicycle: -0.000890486964 PSNR, -0.000001840591 SSIM
```

The regression came from one target-time pairwise `mix0750` decision on bicycle
view `00005`. The source-heldout policy did not need the SSIM tolerance
globally (`source_psnr_dominant_ssim_tolerance_used=false`), so v345e disallows
target-time tolerance in that case.

## Final v345e Focus6 Result

Output root:

```text
outputs/carnet/spcarnet_v345e_source_confirmed_psnr_dominant_pairwise_certificate_focus6_20260702
```

W&B mode:

```text
WANDB_MODE=offline --enable_wandb
```

Machine-readable audit:

```text
docs/car_model/results/v345e_source_confirmed_psnr_dominant_pairwise_focus6_oracle_gap.json
docs/car_model/results/v345e_source_confirmed_psnr_dominant_pairwise_focus6_oracle_gap.md
```

Focus6 selected gains:

| scene | v343e PSNR | v345e PSNR | dPSNR | v343e SSIM | v345e SSIM | dSSIM |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 0.119958548840 | 0.119958548840 | +0.000000000000 | 0.002988750935 | 0.002988750935 | +0.000000000000 |
| bonsai | 0.582901931942 | 0.582901931942 | +0.000000000000 | 0.005913528236 | 0.005913528236 | +0.000000000000 |
| kitchen | 0.493623160533 | 0.493623160533 | +0.000000000000 | 0.003910861697 | 0.003910861697 | +0.000000000000 |
| room | 0.453250185878 | 0.453250185878 | +0.000000000000 | 0.005189244564 | 0.005189244564 | +0.000000000000 |
| stump | 0.058909355358 | 0.058909355358 | +0.000000000000 | 0.001223634928 | 0.001223634928 | +0.000000000000 |
| treehill | 0.116350574542 | 0.118223929370 | +0.001873354828 | 0.001734799809 | 0.001749734084 | +0.000014934275 |

Macro:

| method | PSNR gain | SSIM gain | oracle headroom |
|---|---:|---:|---:|
| v342e | 0.302818959247 | 0.003471774660 | +0.011762124 |
| v343e | 0.304165626182 | 0.003493470028 | +0.010415457 |
| v345e | 0.304477851987 | 0.003495959074 | +0.010103231 |

v345e-v343e:

```text
macro dPSNR: +0.000312225805
macro dSSIM: +0.000002489046
nonnegative PSNR scenes: 6/6
nonnegative SSIM scenes: 6/6
```

Source-confirmed tolerance usage:

```text
treehill: source tolerance used=true; target tolerance used on view 00011, output mix0250
bicycle: source tolerance used=false; target tolerance blocked, no regression
room/bonsai/kitchen/stump: no target tolerance use
```

## Current Interpretation

This is a real and cleaner method step over v343e on the current focus6
support-transport benchmark. It closes one treehill miss, lowers oracle
headroom from `+0.010415457` to `+0.010103231`, and preserves all other focus6
scene metrics under the fixed policy.

It is not a Phase-J breakthrough. The gain remains small and localized, and it
does not solve the older representation-level gap to Phase-J. The next stage
still needs higher-capacity residual transport or a stronger representation
route, not more selector-only threshold layering.

Status:

```text
Final status: NOT COMPLETE.
```
