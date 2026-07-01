# v338 Combined TNC Ranker Negative Reflection

Date: 2026-07-01

## Question Answered

The reflection was useful, but not sufficient for project closure.

It was useful because it moved the work away from blind parameter search and
toward a testable hypothesis: target-neighbor consistency (TNC) should not be a
standalone selector, but may become useful when constrained by source-heldout
local evidence and source-summary safeguards.

It was not sufficient because the resulting v338 combined ranker did not improve
metrics. The validated outcome is a better diagnosis, not a better paper method.

## Implemented Method Change

Main file:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New opt-in interface:

```text
--enable_target_neighbor_combined_candidate_ranker
--target_neighbor_combined_candidate_ranker_feature_grid
--target_neighbor_combined_candidate_ranker_k
--target_neighbor_combined_candidate_ranker_max_target_neighbor_rank
--target_neighbor_combined_candidate_ranker_min_incumbent_minus_candidate_delta
--target_neighbor_combined_candidate_ranker_min_local_psnr_delta_vs_incumbent
--target_neighbor_combined_candidate_ranker_min_local_ssim_delta_vs_incumbent
--target_neighbor_combined_candidate_ranker_min_local_cvar_delta_vs_incumbent
--target_neighbor_combined_candidate_ranker_min_source_summary_psnr_delta_vs_incumbent
--target_neighbor_combined_candidate_ranker_min_source_summary_ssim_delta_vs_incumbent
--target_neighbor_combined_candidate_ranker_require_source_safe
--target_neighbor_combined_candidate_ranker_forbid_fixed_when_incumbent_nonfixed
--target_neighbor_combined_candidate_ranker_enable_ood_guard
```

The ranker computes all-candidate TNC scores without target/test GT and combines
them with source-heldout KNN local evidence. It writes per-view diagnostics,
top-level policy summaries, markdown report sections, and flat W&B metrics. The
existing pure all-candidate TNC diagnostic remains explicitly non-selecting.

## Validation Runs

All runs used W&B offline logging.

```text
outputs/carnet/spcarnet_v338_combined_ranker_smoke3_room_20260701
outputs/carnet/spcarnet_v338_combined_ranker_focus6_20260701
outputs/carnet/spcarnet_v338b_rank1_cvar002_focus6_20260701
```

Saved summaries:

```text
docs/car_model/results/v338_combined_ranker_focus6_oracle_gap.json
docs/car_model/results/v338_combined_ranker_focus6_oracle_gap.md
docs/car_model/results/v338b_combined_ranker_focus6_oracle_gap.json
docs/car_model/results/v338b_combined_ranker_focus6_oracle_gap.md
docs/car_model/results/v338_combined_ranker_focus6_rejection_and_relaxation_diagnostic.json
docs/car_model/results/v338_combined_ranker_focus6_rejection_and_relaxation_diagnostic.md
```

## Actual Focus6 Result

Scenes: `stump, treehill, room, bicycle, bonsai, kitchen`.

| method | scenes | views | selected PSNR gain | selected SSIM gain | promotions |
|---|---:|---:|---:|---:|---:|
| v337diag | 6 | 170 | 0.301231403771 | 0.003460387180 | n/a |
| v338 default | 6 | 170 | 0.301231403771 | 0.003460387180 | 0 |
| v338b rank1/cvar002 | 6 | 170 | 0.301231403771 | 0.003460387180 | 0 |

Oracle headroom is unchanged:

| method | scenes | views | macro headroom |
|---|---:|---:|---:|
| v337diag | 6 | 170 | +0.012506552 |
| v338b | 6 | 170 | +0.012506552 |

This proves the default and rank1/cvar002 combined rankers are no-op policies on
focus6. They are safe, but they do not improve the method.

## Why It Failed

The implementation did not crash. It failed because the policy was too
conservative and because TNC does not correlate strongly enough with true
per-view improvement.

Actual v338b rejection summary:

| scene | views | promote | keep | main candidate reject reasons |
|---|---:|---:|---:|---|
| stump | 16 | 0 | 16 | target_neighbor_rank:59, target_neighbor_margin:5 |
| treehill | 18 | 0 | 18 | target_neighbor_rank:54, fixed_when_incumbent_nonfixed:13, source_local:local_ssim:3 |
| room | 39 | 0 | 39 | target_neighbor_rank:149, fixed_when_incumbent_nonfixed:39, target_neighbor_margin:7 |
| bicycle | 25 | 0 | 25 | target_neighbor_rank:66, fixed_when_incumbent_nonfixed:25, target_neighbor_margin:8 |
| bonsai | 37 | 0 | 37 | target_neighbor_rank:109, fixed_when_incumbent_nonfixed:34, target_neighbor_margin:5 |
| kitchen | 35 | 0 | 35 | target_neighbor_rank:102, fixed_when_incumbent_nonfixed:35, target_neighbor_margin:3 |

Post-hoc relaxed-policy diagnostic over saved v338b reports also rejects this
route. The best posterior relaxed setting gives only:

```text
dPSNR = +0.000240408751
dSSIM = +0.000004195381
promotions = 18
bad promotions = 10
nonnegative PSNR scenes = 5/6
nonnegative SSIM scenes = 5/6
```

The relaxed setting regresses `room` and produces many bad promotions. Because
this relaxed simulation uses target GT only after the fact, it is not a fair
deployable selector. It is useful only as evidence that relaxing the ranker is
not a stable path.

## Reflection Verdict

The reflection did work at the diagnosis level:

1. It confirmed that pure TNC is not a reliable selector.
2. It tested the natural next idea, source-heldout plus TNC combined ranking.
3. It prevented a misleading conclusion: tiny posterior gains from relaxed
   thresholds are not robust research progress.
4. It clarified that the main bottleneck is candidate-generation capacity, not
   rank-threshold tuning.

The reflection did not yet work at the method level:

1. v338 does not beat v337/v336c.
2. It does not reduce oracle headroom.
3. It does not improve qualitative outputs.
4. It should not be promoted as the final method.

## Next Direction

The next real research step should stop treating TNC as a selector and use it as
a training/evidence signal for a stronger candidate generator. A paper-credible
next module should generate a new residual candidate with more capacity than the
current fixed/learned/hybrid/mix/adaptive ladder, while v336c-style source
summary admission and v335-style TNC certificates remain safety layers.

Concrete next prompt:

```text
Implement a target-GT-free support-conditioned residual candidate generator that
uses source-heldout evidence and target-neighbor self-consistency as training or
online regularization, not as a direct ranker. Validate against v336c/v337/v338
on focus6 first, then full9, with oracle-gap, frontier, and qualitative panels.
Do not promote the method unless it improves macro metrics and avoids per-scene
regressions against v336c.
```

## Status

```text
Final status: NOT COMPLETE.
```

v338 is a useful negative milestone and a stronger diagnosis, but not a method
breakthrough.
