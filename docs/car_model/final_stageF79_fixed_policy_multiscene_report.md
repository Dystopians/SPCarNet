# Final Stage F79 Fixed Policy Multiscene Report

Decision: `FIXED_POLICY_MULTISCENE_PASS`.

F79 is the first fixed adaptive-policy run in this branch that passes every tracked metric on all four remaining selected validation scenes. It uses one selector policy and one recovery recipe:

- selector: `csef_adaptive_policy`
- recovery: strict topology freeze, sparse-depth lambda `0.001`, LPIPS lambda `0.00025`
- horizon: `22000 -> 26000`
- topology contract: `--freeze_topology_updates --skip_restricted_delaunay`
- logging: W&B online for every scene

This is not a per-scene parameter table. The selector adapts the prune fraction from checkpoint evidence. The recovery recipe is fixed.

## Why F76 Failed

F76 copied the strong parking F75 behavior too aggressively into smaller scenes. It selected prune fractions around 66% on bonsai, room, and counter, which produced a transfer failure: courtyard passed, but room and counter failed, and bonsai was mixed. That failure was useful because it showed the old rule was still a hidden parking bias.

## Policy Repair

The F79 policy keeps area and local redundancy as the primary signals, but adds stronger global risk control:

- small face-count scenes receive lower maximum prune caps
- acceptable rows are selected under a face-count-dependent risk budget
- if no row clears the risk budget, the fallback uses a narrow minimum-risk band
- the fallback also rejects rows with too much positive-risk evidence

The practical effect is visible in the chosen budgets: F79 keeps courtyard highly compressible at 72%, but moves bonsai to 28.25% and room / counter to 18.5%.

## Quantitative Result

All deltas are method minus the scene-matched clean-long baseline. Positive PSNR / SSIM is better; negative LPIPS / AbsRel / Depth MAE / Normal is better.

| scene | W&B | adaptive prune | triangles | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bonsai | `6y0kyntt` | 0.282500 | 63,470 | 28.3% | +0.121841 | +0.018046 | -0.013154 | -0.012211 | -0.011805 | -3.129279 | `PASS_ALL_METRIC_CLEAN_WIN` |
| courtyard | `xv4xvi32` | 0.720000 | 469,696 | 72.0% | +0.103273 | +0.012519 | -0.002680 | -0.052788 | -0.492201 | -0.507621 | `PASS_ALL_METRIC_CLEAN_WIN` |
| room | `pcde5er3` | 0.185000 | 68,872 | 18.5% | +0.860854 | +0.082661 | -0.061458 | -0.016006 | -0.085100 | -1.467686 | `PASS_ALL_METRIC_CLEAN_WIN` |
| counter | `fgdue0tb` | 0.185000 | 68,325 | 18.5% | +0.261847 | +0.030848 | -0.024286 | -0.006674 | -0.018694 | -1.118089 | `PASS_ALL_METRIC_CLEAN_WIN` |

Summary: available rows `4 / 4`; all-metric clean wins `4 / 4`.

## Evidence Files

- `outputs/carnet/meshsplatopt/final_stageF79_fixed_adaptive_policy_multiscene/fixed_adaptive_policy_multiscene_results.md`
- `outputs/carnet/meshsplatopt/final_stageF79_fixed_adaptive_policy_multiscene/fixed_adaptive_policy_multiscene_results.json`
- `outputs/carnet/meshsplatopt/final_stageF79_fixed_adaptive_policy_multiscene/fixed_adaptive_policy_multiscene_results.csv`

## Reflection

The main lesson is that "adaptive" cannot mean selecting the most aggressive successful parking-like fraction. It has to reason about scene scale and risk. F79 is materially better than the earlier validation-budget table because it removes the need for manual prune ratios on the four multiscene validation scenes while preserving the same fairness standard: clean-long baseline versus long-recovery method at matched scene data and independent render / geometry evaluation.

The result supports the selected-scene claim. It should not be written as a universal claim over all possible scenes until more datasets are added.
