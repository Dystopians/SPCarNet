# v336c Adaptive Residual Candidate With Source-Summary Gate

Date: 2026-07-01

## Motivation

v335 was safe and positive, but its gain was concentrated in treehill. v336 tried
to add a real generated candidate instead of only changing arbitration:

- `adaptive`: a per-pixel blend between the fixed source evidence residual and
  the learned calibrator residual;
- `tnc_gen`: a target-neighbor render/depth/camera consensus candidate.

The target-neighbor generated candidate was rejected after focused3 testing
because it was strongly negative on bonsai and room. The adaptive candidate was
more promising, but the first v336b full9 run exposed a policy-level failure:
adding `adaptive` into the source-reliability candidate pool improved room but
slightly regressed garden. The issue was not target GT leakage; it was candidate
pool perturbation. A weak generated candidate changed the learned source
reliability model's decisions for existing strong candidates.

v336c fixes that with an automatic source-heldout safety gate. Generated
candidates are allowed into downstream policy fitting only when their
source-heldout scene summary is safe versus the scene incumbent. Otherwise they
are removed from the selector payload before policy fitting. This is scene-name
free, target-GT-free, and keeps the original v335 policy behavior when the new
candidate lacks source evidence.

## Implementation

Main code:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New/extended interfaces:

```text
--enable_adaptive_residual_candidate
--adaptive_residual_candidate_name adaptive
--adaptive_residual_max_blend 0.75
--generated_candidate_scene_selection_mode candidate_only
--generated_candidate_disable_when_scene_fixed
--generated_candidate_require_source_summary_safe
--generated_candidate_min_source_summary_psnr_delta_vs_scene 0.0
--generated_candidate_min_source_summary_ssim_delta_vs_scene -0.00005
```

The adaptive residual gate combines source evidence confidence, support count,
residual stability, and fixed/learned residual alignment. The source-summary
gate then checks the generated candidate's source-heldout aggregate before it is
allowed to influence source-reliability, KNN, local-support, risk, or pairwise
policies. Suppression reasons are written into each report under:

```text
policy.suppressed_generated_candidate_reasons
```

Engineering fixes in the same milestone:

- generated candidates can be filtered out of selector payloads without causing
  downstream `KeyError`;
- policy fit functions now respect `selector_payload["candidate_variants"]`;
- `online_target_proxy_enabled` now checks the actual
  `enable_target_neighbor_consistency_certificate` flag.

## Validation

All medium/full runs used W&B offline logging and explicit GPU assignment.

Key output roots:

```text
outputs/carnet/spcarnet_v336b_adaptive_fixedguard_focused3_rerun_20260701
outputs/carnet/spcarnet_v336b_adaptive_fixedguard_full9_20260701
outputs/carnet/spcarnet_v336c_source_summary_gate_probe_20260701
outputs/carnet/spcarnet_v336c_source_summary_gate_full9_20260701
outputs/carnet/spcarnet_v336c_frontier_full9_20260701
```

Archived summaries:

```text
docs/car_model/results/v336c_source_summary_gate_full9_vs_v335_v336b_audit.json
docs/car_model/results/v336c_source_summary_gate_full9_vs_v335_v336b_audit.md
docs/car_model/results/v336c_frontier_lpips_qualitative_summary.json
docs/car_model/results/v336c_frontier_lpips_qualitative_summary.md
docs/car_model/results/v336c_frontier_panels/
```

## Full9 Apply Metrics

| method | scenes | selected PSNR gain | selected SSIM gain | all-axis safe |
|---|---:|---:|---:|---:|
| v335 | 9 | 0.274017908934 | 0.003741526179 | 9/9 |
| v336b | 9 | 0.274583943273 | 0.003745085387 | 9/9 |
| v336c | 9 | 0.274617423486 | 0.003744976625 | 9/9 |

v336b improved macro metrics but regressed garden. v336c fixes that:

| comparison | dPSNR | dSSIM | nonnegative PSNR scenes | nonnegative SSIM scenes |
|---|---:|---:|---:|---:|
| v336c - v335 | +0.000599514552 | +0.000003450447 | 9/9 | 9/9 |
| v336c - v336b | +0.000033480213 | -0.000000108762 | 7/9 | 7/9 |

Per-scene v336c vs v335:

| scene | dPSNR | dSSIM | generated active | reason |
|---|---:|---:|---|---|
| bicycle | +0.000000000000 | +0.000000000000 | false | source summary unsafe |
| bonsai | +0.000000000000 | +0.000000000000 | false | source summary unsafe |
| counter | +0.000000000000 | +0.000000000000 | false | source summary unsafe |
| flowers | +0.000000000000 | +0.000000000000 | false | source summary unsafe |
| garden | +0.000000000000 | +0.000000000000 | false | source summary unsafe |
| kitchen | +0.000000000000 | +0.000000000000 | false | source summary unsafe |
| room | +0.005395630969 | +0.000031054020 | true | adaptive accepted |
| stump | +0.000000000000 | +0.000000000000 | false | scene selected fixed |
| treehill | +0.000000000000 | +0.000000000000 | false | scene selected fixed |

## Full9 Frontier Metrics

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v335 | 27.590394 | 0.028168 | 0.087742 | 0.057670 |
| v336b | 27.590928 | 0.028167 | 0.087737 | 0.057667 |
| v336c | 27.590966 | 0.028167 | 0.087738 | 0.057667 |

v336c is better than clean26000 by `+0.397323` PSNR, `-0.000945` MAE,
`-0.002469` LPIPS, and `-0.002235` DISTS. Compared with v335, it improves
PSNR/MAE/DISTS and keeps LPIPS effectively in the same narrow band. Compared
with v336b, it trades a negligible LPIPS loss for removing the garden regression.

## Qualitative Evidence

Panel examples were generated for room, garden, treehill, and bonsai:

```text
docs/car_model/results/v336c_frontier_panels/room/00004_frontier_panel.png
docs/car_model/results/v336c_frontier_panels/room/00009_frontier_panel.png
docs/car_model/results/v336c_frontier_panels/garden/00006_frontier_panel.png
docs/car_model/results/v336c_frontier_panels/garden/00017_frontier_panel.png
docs/car_model/results/v336c_frontier_panels/treehill/00001_frontier_panel.png
docs/car_model/results/v336c_frontier_panels/treehill/00011_frontier_panel.png
docs/car_model/results/v336c_frontier_panels/bonsai/00001_frontier_panel.png
docs/car_model/results/v336c_frontier_panels/bonsai/00035_frontier_panel.png
```

The most important qualitative interpretation is not that v336c creates a large
visible jump everywhere. It preserves v335 where adaptive evidence is weak, and
only activates on room where source-heldout evidence supports the adaptive
candidate. This is a reliability improvement over v336b, not a final visual
breakthrough.

## Lessons

1. Generated candidates must not be injected blindly into a learned policy pool.
   Even a plausible candidate can perturb decisions among older strong
   candidates.
2. Source-heldout aggregate evidence is useful as an admission certificate:
   it can preserve non-regression without using target/test GT.
3. The current adaptive residual is a real candidate-generation mechanism, but
   its coverage is narrow. In full9 it is admitted only on room.
4. `tnc_gen` is not ready. Its focused3 result was negative and it should remain
   disabled unless redesigned.

## Status

```text
Final status: NOT COMPLETE.
```

v336c is a verified positive milestone over v335 and fixes the v336b regression,
but it is still not a paper-final method. The gain is small and mostly comes
from one scene. The next research step should improve generated residual
capacity itself, while retaining v336c-style source-summary admission as a
safety layer.
