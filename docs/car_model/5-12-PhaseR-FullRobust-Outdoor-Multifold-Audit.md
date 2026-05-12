# Phase-R Full-Robust Outdoor Multi-Fold Audit

Date: 2026-05-12

## Purpose

This audit closes a fairness gap in the earlier Phase-R v10 snapshot.  v10
mixed strict indoor multi-fold decisions with legacy single train-val outdoor
decisions.  That was useful for exploration, but it was too optimistic for a
paper-facing representation-level claim.  I therefore reran the outdoor
surface-SH1 candidates with the same four-offset train-only gate already used
for indoor scenes.

The held-out test split remains report-only.  No test metric is used for
candidate selection, gamma selection, fallback, or threshold changes.

## Interface Change

`scripts/car_model/ecsr_select_phase_r_policy.py` now rejects any decision file
that declares `selection_uses_test=true`, even if the decision also says
`accepted=true`.  Current decisions already set this flag to false, so this is
a policy hardening change rather than a result-changing patch.

## Outdoor Full-Strength Multi-Fold Results

Thresholds: min PSNR gain `0.0`, max SSIM regression `5e-5`, max LPIPS
regression `1.5e-4`.

| scene | candidate | decision | mean dPSNR | mean dSSIM | mean dLPIPS | rejection reason | report-only test dPSNR | dSSIM | dLPIPS |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| bicycle | SH1 full | reject | +0.000081 | -0.000062 | -0.000027 | offset1 PSNR/SSIM, offset3 SSIM | +0.001156 | +0.000135 | -0.000432 |
| flowers | SH1 full | reject | +0.000218 | -0.000015 | +0.000084 | offset1 LPIPS, offset3 SSIM | +0.002346 | +0.000344 | -0.000405 |
| garden | SH1 full | reject | +0.000183 | -0.000016 | +0.000054 | offset1 PSNR/SSIM | +0.000662 | +0.000024 | -0.000036 |
| stump | SH1 full | accept | +0.000102 | -0.000003 | +0.000005 | pass | +0.000021 | +0.000000 | -0.000011 |

The important correction is that several outdoor candidates that looked fine
under a single split are not robust under complementary train-heldout offsets.
They can remain diagnostic candidates, but they are no longer promoted into the
fixed Phase-R policy.

## Gamma 0.25 Trust-Region Negative Control

I also materialized a fixed, non scene-tuned gamma trust-region candidate for
the same outdoor edits:

`source + 0.25 * (candidate - source)`

This was tested because `room` benefited from a train-only trust-region blend.
The same idea does not rescue the outdoor failures.

| scene | gamma candidate | decision | mean dPSNR | mean dSSIM | mean dLPIPS | rejection reason |
|---|---|---:|---:|---:|---:|---|
| bicycle | gamma 0.25 | reject | +0.000141 | +0.000018 | +0.000063 | offset3 LPIPS |
| flowers | gamma 0.25 | reject | +0.000066 | -0.000009 | +0.000100 | offset1 LPIPS |
| garden | gamma 0.25 | reject | -0.000003 | -0.000009 | +0.000037 | offset0/1 PSNR |
| stump | gamma 0.25 | accept | +0.000016 | -0.000000 | +0.000000 | pass |

`stump` already passed at full strength, and the gamma version is weaker, so
the fixed ladder keeps the full-strength stump edit.  The gamma rows mainly
serve as negative evidence: simply shrinking the residual is not enough to make
`bicycle`, `flowers`, or `garden` paper-clean.

## v11 Fixed Ladder

Artifact:

`outputs/carnet/meshsplatopt/ecsr_phase_r/fixed_candidate_ladder_v11_fullrobust_alloffset/phase_r_fixed_candidate_ladder.md`

Result:

- scenes: `9`
- train-val accepted representation edits: `3 / 9`
- accepted scenes: `stump`, `room`, `kitchen`
- fallback scenes: `bicycle`, `flowers`, `garden`, `treehill`, `counter`, `bonsai`
- report-only strict RGB wins among selected edits: `3 / 9`
- mean report-only delta vs Phase-J with no-op fallback: PSNR `+0.002531`, SSIM
  `+0.000080`, LPIPS `-0.000120`

| scene | v11 selected | gate | accepted | mean train-val dPSNR | dSSIM | dLPIPS | report-only test dPSNR | dSSIM | dLPIPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | fallback | none | false | 0 | 0 | 0 | 0 | 0 | 0 |
| flowers | fallback | none | false | 0 | 0 | 0 | 0 | 0 | 0 |
| garden | fallback | none | false | 0 | 0 | 0 | 0 | 0 | 0 |
| stump | outdoor full SH1 | multi-fold | true | +0.000102 | -0.000003 | +0.000005 | +0.000021 | +0.000000 | -0.000011 |
| treehill | fallback | predeclared | false | 0 | 0 | 0 | 0 | 0 | 0 |
| room | gamma trust SH1 | multi-fold | true | +0.000089 | -0.000002 | +0.000000 | +0.000084 | +0.000001 | -0.000000 |
| counter | fallback | none | false | 0 | 0 | 0 | 0 | 0 | 0 |
| kitchen | sparse SH1 | multi-fold | true | +0.000537 | +0.000014 | -0.000011 | +0.022673 | +0.000719 | -0.001068 |
| bonsai | fallback | none | false | 0 | 0 | 0 | 0 | 0 | 0 |

## Interpretation

This is a reliability upgrade, not a visual breakthrough.  The previous v10
snapshot overstated Phase-R completion because some outdoor positives were
single-split accepts.  The full-robust v11 policy is less impressive on the
headline acceptance count, but it is much more defensible: every selected
representation edit passes the same multi-offset train-only certificate.

The scientific lesson is sharp:

- surface-attached SH1 residuals are real and can be checkpoint-baked;
- they generalize cleanly on `stump`, `room`, and especially `kitchen`;
- `bicycle`, `flowers`, `garden`, `counter`, and `bonsai` need a different
  operator, not weaker thresholds or post-hoc gamma shrinking;
- Phase-J remains the paper-facing RGB endpoint for now, while Phase-R is the
  rigorous representation-level baseline and diagnostic path.

## Next Required Work

The next representation-level attempt should not be another residual-strength
scan.  The failures point to operator mismatch:

1. `flowers` and `bicycle` fail on perceptual or SSIM tails despite positive
   mean/test deltas, so they likely need local view-dependent capacity or a
   stricter surface support ownership model.
2. `garden` fails PSNR under gamma and full strength, so the current SH1 write
   direction is locally wrong on some train-heldout offsets.
3. `bonsai` has systematic offset-3 LPIPS regression, so accepting it by
   relaxing thresholds would be parameter gaming.
4. `counter` remains an indoor generalization failure; its micro residual
   passed one split but failed multi-fold and test.

The method should therefore move toward certificate-carrying local surface
codes or contraction-aware appearance relocation, while keeping this v11
multi-offset gate as the minimum acceptance standard.
