# v339 TNC-Regularized Residual Candidate

Date: 2026-07-01

## Purpose

v338 showed that another target-neighbor ranker is not enough. The next test was
therefore a real generated candidate, not another selector threshold.

v339 adds `tnc_reg`, a target-GT-free residual candidate. It uses
target-neighbor render/depth/camera self-consistency as a weak regularizer for
the residual itself:

1. start from a source-supported base residual (`adaptive` in the fair run);
2. compare fixed and learned residuals against target-neighbor consensus;
3. move per pixel only through a source-evidence reliability gate;
4. keep the existing source-summary admission gate before the candidate can
   enter downstream policies.

It does not copy target-neighbor consensus directly into the output.

## Implementation

Main file:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New interface:

```text
--enable_tnc_regularized_residual_candidate
--tnc_regularized_residual_candidate_name tnc_reg
--tnc_regularized_base_variant
--tnc_regularized_learned_variant
--tnc_regularized_fixed_variant
--tnc_regularized_max_learned_step
--tnc_regularized_max_fixed_step
--tnc_regularized_agreement_scale
--tnc_regularized_decision_temperature
--tnc_regularized_learned_margin
--tnc_regularized_delta_clip
--tnc_regularized_min_consensus_effective_weight
--tnc_regularized_min_confident_fraction
```

The tested conservative defaults are:

```text
base_variant=adaptive
learned_margin=0.001
max_learned_step=0.15
max_fixed_step=0.20
```

The apply report now records:

```text
policy.tnc_regularized_residual_candidate
per_view[*].target_neighbor_generated_candidate_diagnostics.regularized_residual
```

## Validation Roots

Smoke/probes:

```text
outputs/carnet/spcarnet_v339_tnc_regularized_smoke3_room_20260701
outputs/carnet/spcarnet_v339b_tnc_reg_adaptive_smoke3_room_20260701
outputs/carnet/spcarnet_v339c_tnc_reg_margin_smoke3_room_20260701
outputs/carnet/spcarnet_v339e_tnc_reg_learnedbase_room_probe_20260701
```

Focus6:

```text
outputs/carnet/spcarnet_v339c_tnc_reg_margin_focus6_20260701
outputs/carnet/spcarnet_v339d_tnc_reg_fullstack_focus6_20260701
```

Saved summaries:

```text
docs/car_model/results/v339_tnc_regularized_residual_focus6_summary.json
docs/car_model/results/v339_tnc_regularized_residual_focus6_summary.md
docs/car_model/results/v339d_tnc_reg_fullstack_focus6_oracle_gap.json
docs/car_model/results/v339d_tnc_reg_fullstack_focus6_oracle_gap.md
```

All apply/eval runs used W&B offline logging.

## Fair Focus6 Result

The fair run is v339d: complete v337/v335/v334/v333 stack plus `adaptive` and
`tnc_reg`, with source-summary admission enabled.

| method | scenes | views | selected PSNR gain | selected SSIM gain | oracle headroom |
|---|---:|---:|---:|---:|---:|
| v337diag | 6 | 170 | 0.301231403771 | 0.003460387180 | +0.012506552 |
| v339d full stack | 6 | 170 | 0.301231403771 | 0.003460387180 | +0.012506552 |

Per scene, v339d exactly matches v337diag:

| scene | PSNR gain | SSIM gain | selected | generated-candidate outcome |
|---|---:|---:|---|---|
| stump | 0.057029761393 | 0.001208242029 | fixed | adaptive/tnc_reg rejected by source-summary SSIM |
| treehill | 0.118121382508 | 0.001717434989 | fixed | adaptive/tnc_reg rejected by source-summary SSIM |
| room | 0.442681127076 | 0.005089075137 | hybrid | tnc_reg active, but weaker than existing selected behavior |
| bicycle | 0.119958548840 | 0.002988750935 | hybrid | adaptive/tnc_reg rejected by source-summary PSNR |
| bonsai | 0.575974442276 | 0.005847958294 | learned | adaptive/tnc_reg rejected by source-summary PSNR |
| kitchen | 0.493623160533 | 0.003910861697 | learned | adaptive/tnc_reg rejected by source-summary PSNR |

## Probe Lessons

The first smoke with only `tnc_reg` did not test the intended method because the
candidate was rejected by source-summary evidence. It also exposed that a small
3-view room subset can make source-heldout scene selection look worse than
fixed, so smoke metrics should not be used as a claim.

The adaptive-base smoke was valid:

| candidate | room smoke3 PSNR gain | room smoke3 SSIM gain |
|---|---:|---:|
| adaptive | 0.256933644933 | 0.003750622272 |
| tnc_reg default | 0.255735866259 | 0.003743469715 |
| tnc_reg conservative | 0.256171303694 | 0.003743807475 |

The conservative margin made `tnc_reg` less harmful, but it still did not beat
`adaptive`.

The learned-base shrink probe was also negative. It was rejected before target
selection:

```text
tnc_reg: source_summary_psnr_delta:-0.0135018206
```

This means target-neighbor consensus did not provide a useful source-heldout
residual shrink candidate on room.

## Interpretation

v339 is a real engineering/method change, but not a positive method result.

What worked:

- The pipeline now supports a target-GT-free TNC-regularized residual candidate.
- The candidate is audited through source-heldout admission and per-view
  diagnostics.
- The full-stack run is non-regressive because unsafe generated candidates are
  filtered out.

What did not work:

- The candidate only survives admission on room.
- On room, `tnc_reg` is weaker than the existing adaptive and selected outputs.
- It does not reduce oracle headroom.
- It does not move focus6 macro metrics beyond v337diag.

## Updated Bottleneck

The bottleneck is now sharper:

1. Target-neighbor consistency is useful for rollback/unlock certificates.
2. It is not strong enough as a direct ranker.
3. It is also not strong enough as a hand-crafted residual generator.
4. The next candidate generator likely needs learned capacity trained on
   source-heldout residual outcomes, with TNC used as an auxiliary consistency
   feature/loss rather than a direct formula.

## Status

```text
Final status: NOT COMPLETE.
```

v339 should be kept as a documented negative/diagnostic milestone. It is not a
promoted paper method.
