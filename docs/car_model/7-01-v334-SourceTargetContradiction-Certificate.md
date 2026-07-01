# v334 Source-Target Contradiction Certificate

Date: 2026-07-01

## Purpose

v333 introduced target-neighbor render self-consistency and correctly rolled
back treehill `00007` and `00008`, but still missed `00009`. The diagnostic
failure mode was specific: source-local pairwise evidence was very confident,
yet target-neighbor consistency mildly preferred the incumbent. v334 turns this
into an explicit target-blind certificate instead of another parameter sweep.

## Implemented Interfaces

Main pipeline:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New opt-in CLI:

```text
--target_neighbor_consistency_enable_source_contradiction
--target_neighbor_consistency_contradiction_min_source_local_min_delta
--target_neighbor_consistency_contradiction_min_source_local_cvar_delta
--target_neighbor_consistency_contradiction_min_source_positive_fraction
--target_neighbor_consistency_contradiction_max_incumbent_minus_output_delta
```

The new branch is disabled by default. It only runs when the existing
target-neighbor consistency certificate is enabled.

## Mechanism

For a promoted pairwise candidate, v334 first runs the v333 target-neighbor
consistency check. If the normal rollback threshold is not met, v334 applies a
second contradiction test:

1. source-local pairwise support must be strongly positive;
2. source-local PSNR min and CVaR must both exceed the frozen threshold;
3. source-local positive fraction must meet the frozen threshold;
4. target-neighbor consistency must mildly prefer the incumbent.

With the frozen full9 policy, the contradiction thresholds are:

```text
source psnr_min >= 0.01
source psnr_cvar >= 0.01
source positive fraction >= 1.0
incumbent_minus_output_mae_delta < -0.00002
normal target-neighbor rollback threshold < -0.0001
```

This is still target/test-GT-free at decision time. Target/test GT is read only
after images are saved for evaluation.

## Focused Treehill Result

Output:

```text
outputs/carnet/spcarnet_v334_source_target_contradiction_treehill_20260701
docs/car_model/results/v334_source_target_contradiction_treehill_report.json
```

W&B offline run:

```text
outputs/carnet/spcarnet_v334_source_target_contradiction_treehill_20260701/wandb/offline-run-20260701_040116-nxfl8s6p
```

| method | selected PSNR gain | selected SSIM gain | rollback count |
|---|---:|---:|---:|
| v333 treehill | 0.106409362285 | 0.001693874598 | 2 |
| v334 treehill | 0.107097397630 | 0.001694096459 | 3 |
| delta | +0.000688035345 | +0.000000221862 | +1 |

The new rollback is treehill `00009`, with reason
`source_target_neighbor_contradiction`. Positive controls `00011` and `00015`
remain promoted.

## Full9 Replay

Full output root:

```text
outputs/carnet/spcarnet_v334_source_target_contradiction_full9_20260701
```

Audit files:

```text
docs/car_model/results/v334_source_target_contradiction_full9_vs_v333_v329b_audit.json
docs/car_model/results/v334_source_target_contradiction_full9_vs_v333_v329b_audit.md
```

Macro:

| metric | v329b | v333 | v334 | v334-v333 | v334-v329b |
|---|---:|---:|---:|---:|---:|
| selected PSNR gain | 0.272522652479 | 0.272716573354 | 0.272793021725 | +0.000076448372 | +0.000270369246 |
| selected SSIM gain | 0.003736660673 | 0.003738908357 | 0.003738933009 | +0.000000024651 | +0.000002272335 |
| rollback count | 0 | 2 | 3 | +1 | +3 |
| all-axis safe scenes | 9/9 | 9/9 | 9/9 |  |  |

The only scene changed relative to v333 is treehill.

Treehill critical views:

| view | raw variant | final variant | rollback | reason | target-neighbor delta | source psnr_min | raw-vs-fixed PSNR delta |
|---|---|---|---:|---|---:|---:|---:|
| 00007 | mix0250 | fixed | true | target_neighbor_consistency_delta | -0.000121739321 | 0.021059423675 | -0.026469408413 |
| 00008 | mix0250 | fixed | true | target_neighbor_consistency_delta | -0.000148012727 | 0.000413699791 | -0.004945773279 |
| 00009 | mix0250 | fixed | true | source_target_neighbor_contradiction | -0.000024989738 | 0.021059423675 | -0.012384636218 |
| 00011 | mix0250 | mix0250 | false | passed | -0.000070053628 | 0.000413699791 | +0.033720386903 |
| 00015 | mix0250 | mix0250 | false | passed | -0.000044425695 | 0.000413699791 | +0.015076869936 |

## Perceptual and Qualitative Frontier

Output:

```text
outputs/carnet/spcarnet_v334_frontier_comparison_full9_20260701
docs/car_model/results/v334_frontier_lpips_qualitative_summary.json
docs/car_model/results/v334_frontier_lpips_qualitative_summary.md
docs/car_model/results/v334_frontier_panels/
```

W&B offline run:

```text
outputs/carnet/spcarnet_v334_frontier_comparison_full9_20260701/wandb/offline-run-20260701_041135-jshmpiav
```

Aggregate:

| method | scenes | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|
| clean26000 | 9 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v329b | 9 | 27.588444 | 0.028173 | 0.087733 | 0.057664 |
| v333 | 9 | 27.588734 | 0.028171 | 0.087735 | 0.057664 |
| v334 | 9 | 27.588834 | 0.028170 | 0.087735 | 0.057664 |

Dedicated panel:

```text
docs/car_model/results/v334_frontier_panels/treehill_00009_v333_v334_gt_panel.png
```

Reading: v334 does not regress LPIPS/DISTS and slightly improves PSNR/MAE over
v333, but the visual difference is subtle. The panel is useful for explaining
the decision logic, not for claiming a strong qualitative leap.

## Reflection

The reflection did help at v334 because it moved the work from scalar threshold
tuning to failure-mode modeling. The missed case was not "low source support";
it was "source evidence is too confident while target-neighbor evidence quietly
disagrees." Encoding that contradiction caught the remaining treehill `00009`
case without dropping `00011` and `00015`.

The limitation is equally clear: the improvement is a narrow safety certificate
on top of the existing candidate generator. It is not yet a representation-level
breakthrough, and it does not make the qualitative advantage obvious to a human
viewer. The next real leap likely requires stronger candidate generation or a
new image/geometry representation module, not more rollback certificates.

Status:

```text
Final status: NOT COMPLETE.
```
