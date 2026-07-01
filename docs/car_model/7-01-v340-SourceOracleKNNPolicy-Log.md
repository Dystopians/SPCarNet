# v340 Source-Oracle KNN Policy and Reliability Agreement Log

Date: 2026-07-01

Status: implemented and focus6-validated, but not paper-level closure.

## Motivation

The v337-v339 line showed that target-neighbor consistency (TNC) is useful as a
diagnostic/certificate, but not as a direct selector or hand-written residual
generator. Pure TNC ranking was poorly aligned with the strict oracle, v338's
combined TNC ranker became a safe no-op, and v339's TNC-regularized residual
candidate was rejected or failed to beat the existing adaptive candidate.

The v340 direction therefore changes the decision mechanism. Instead of using a
manual TNC residual rule, it learns a target-GT-free per-view candidate policy
from source-heldout oracle outcomes:

- source-heldout views provide candidate metrics after outputs are fixed;
- candidate/proxy features are built without target/test GT;
- leave-one-out source-heldout KNN votes for the candidate that wins on similar
  source validation views;
- target/test application uses only source-heldout policy state and target-blind
  candidate proxies.

## Implementation

Main file:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New policy:

```text
--enable_source_oracle_knn_policy
--source_oracle_knn_apply_mode {pre_reliability,post_reliability_scene_only}
--source_oracle_knn_require_reliability_agreement
```

Core helpers:

```text
_source_oracle_knn_view_feature
_source_oracle_knn_best_variant
_source_oracle_knn_vote
_source_oracle_knn_neighbor_summary
_fit_source_oracle_knn_policy
_source_oracle_knn_choose_variant
```

The promoted path is `post_reliability_scene_only`: source-reliability makes the
first decision, and source-oracle KNN is only allowed to propose a promotion
when source-reliability leaves the view at the scene incumbent. This avoids the
v340a failure where KNN overrode already good source-reliability choices.

v340d adds a second agreement gate. A source-oracle KNN promotion is accepted
only if the source-reliability predictor also estimates nonnegative PSNR and
SSIM deltas versus the scene incumbent:

```text
--source_oracle_knn_min_reliability_predicted_psnr_delta 0.0
--source_oracle_knn_min_reliability_predicted_ssim_delta 0.0
```

This is not target tuning: the agreement check uses the already fitted
source-heldout reliability predictor before target/test GT is read.

## Static Validation

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/apply_source_heldout_support_transport_calibrator.py

git diff --check -- scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

Both passed before the full focus6 runs.

## Experiment Roots

Exploratory:

```text
outputs/carnet/spcarnet_v340_source_oracle_knn_smoke3_room_20260701
outputs/carnet/spcarnet_v340a_source_oracle_knn_room_20260701
outputs/carnet/spcarnet_v340b_source_oracle_knn_postfallback_room_20260701
outputs/carnet/spcarnet_v340b_source_oracle_knn_postfallback_focus6_20260701
outputs/carnet/spcarnet_v340c_source_oracle_agreement_room_20260701
outputs/carnet/spcarnet_v340c_source_oracle_agreement_focus6_20260701
```

Promoted fair focus6 replay:

```text
outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701
```

Comparison reports:

```text
docs/car_model/results/v340b_source_oracle_knn_postfallback_focus6_oracle_gap.json
docs/car_model/results/v340b_source_oracle_knn_postfallback_focus6_oracle_gap.md
docs/car_model/results/v340c_source_oracle_agreement_focus6_oracle_gap.json
docs/car_model/results/v340c_source_oracle_agreement_focus6_oracle_gap.md
docs/car_model/results/v340d_source_oracle_agreement_pairwise_focus6_oracle_gap.json
docs/car_model/results/v340d_source_oracle_agreement_pairwise_focus6_oracle_gap.md
```

v340d W&B offline runs:

```text
bicycle:  outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701/bicycle/wandb/offline-run-20260701_075422-dejvnq9g
bonsai:   outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701/bonsai/wandb/offline-run-20260701_075545-vrhs2bic
kitchen:  outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701/kitchen/wandb/offline-run-20260701_075532-eg7gw08n
room:     outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701/room/wandb/offline-run-20260701_075546-nw5h21v2
stump:    outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701/stump/wandb/offline-run-20260701_075358-ytgfzo4s
treehill: outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701/treehill/wandb/offline-run-20260701_075418-ve03q9go
```

## Key Results

Focus6 scenes:

```text
stump, treehill, room, bicycle, bonsai, kitchen
```

Macro comparison:

| method | selected PSNR gain | selected SSIM gain | PSNR vs v337 | SSIM vs v337 |
|---|---:|---:|---:|---:|
| v337diag | 0.301231403771 | 0.003460387180 | 0.000000000000 | 0.000000000000 |
| v340b oracle KNN | 0.301720711625 | 0.003458363934 | +0.000489307854 | -0.000002023247 |
| v340d agreement + pairwise | 0.301510278428 | 0.003461709518 | +0.000278874657 | +0.000001322338 |

Per-scene comparison:

| scene | v337 PSNR | v340b PSNR | v340d PSNR | v340d-v337 | v337 SSIM | v340b SSIM | v340d SSIM | v340d-v337 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stump | 0.057029761393 | 0.057029761393 | 0.057029761393 | +0.000000000000 | 0.001208242029 | 0.001208242029 | 0.001208242029 | +0.000000000000 |
| treehill | 0.118121382508 | 0.118121382508 | 0.118121382508 | +0.000000000000 | 0.001717434989 | 0.001717434989 | 0.001717434989 | +0.000000000000 |
| room | 0.442681127076 | 0.445616974198 | 0.444247548761 | +0.001566421685 | 0.005089075137 | 0.005076935658 | 0.005089608523 | +0.000000533385 |
| bicycle | 0.119958548840 | 0.119958548840 | 0.119958548840 | +0.000000000000 | 0.002988750935 | 0.002988750935 | 0.002988750935 | +0.000000000000 |
| bonsai | 0.575974442276 | 0.575974442276 | 0.576081268530 | +0.000106826254 | 0.005847958294 | 0.005847958294 | 0.005855358936 | +0.000007400642 |
| kitchen | 0.493623160533 | 0.493623160533 | 0.493623160533 | +0.000000000000 | 0.003910861697 | 0.003910861697 | 0.003910861697 | +0.000000000000 |

v340b had the larger PSNR gain, but it regressed macro SSIM and hurt room SSIM.
v340d gives up part of that PSNR gain to recover SSIM and becomes a small
all-axis improvement over v337diag.

## Important Experiment Lesson

The first v340c focus6 run looked worse than v337 in PSNR. That was not a method
failure; it was an unfair command mismatch. The run omitted the pairwise
dominance and promotion-rollback policy used by v337/v340b on treehill, causing
treehill to lose 4 `mix0250` pairwise decisions and regress in PSNR.

The corrected v340d replay restores the pairwise/promotion-rollback path and
recovers treehill exactly to v337/v340b:

```text
treehill selected PSNR gain: 0.118121382508
treehill selected SSIM gain: 0.001717434989
```

This is a concrete process correction: policy comparisons must preserve the full
frozen decision stack unless the ablation explicitly removes a module.

## Reflection Verdict

The reflection was useful but not sufficient.

It was useful because it changed the work from blind threshold scanning to a
more defensible source-heldout outcome policy with an independent reliability
agreement certificate. It also caught a false negative caused by an unfair
command mismatch.

It is not sufficient because the resulting v340d gain is still small:

```text
macro PSNR gain over v337: +0.000278874657
macro SSIM gain over v337: +0.000001322338
```

This is a real method change and a verified all-axis focus6 milestone, but not
the kind of large, visually obvious improvement needed for a top-conference
paper endpoint.

## Remaining Weaknesses

- v340d does not beat v340b on PSNR; it trades PSNR for SSIM safety.
- Gains concentrate in room and bonsai; stump, treehill, bicycle, and kitchen
  are unchanged.
- Oracle headroom is not reduced meaningfully: v340d macro headroom is
  `+0.012505689`, essentially the same as v337's `+0.012506552`.
- The source-oracle KNN policy remains a selector over existing candidates; it
  does not create a stronger representation or residual basis.
- Qualitative gains are expected to remain subtle because only a few views are
  changed and the average metric delta is small.

## Next Step

The next credible improvement should target candidate generation rather than
another selector layer. The current evidence says arbitration can recover small
safe gains, but the remaining oracle headroom requires stronger candidates:

- a source-heldout outcome model that predicts shrink/blend magnitude, not only
  discrete candidate identity;
- a richer generated residual candidate trained/evaluated on source-heldout
  outcomes;
- explicit ablations separating candidate capacity from safety certificates.

Final status: NOT COMPLETE.
