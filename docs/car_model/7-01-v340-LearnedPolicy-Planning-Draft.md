# v340 Learned Source-Heldout Residual Outcome Policy Planning Draft

Date: 2026-07-01

Status: superseded by the implemented v340 source-oracle KNN policy log.

Implementation and experiments are now recorded in:

```text
docs/car_model/7-01-v340-SourceOracleKNNPolicy-Log.md
docs/car_model/results/v340d_source_oracle_agreement_pairwise_focus6_oracle_gap.md
outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701
```

The implemented version is a conservative first step toward the policy proposed
here: it learns a source-heldout oracle-neighborhood candidate selector and adds
a source-reliability agreement certificate. It does not yet implement a richer
outcome model that predicts continuous shrink/blend magnitude.

## Purpose

This draft records why the v338/v339 evidence says manual TNC residual logic is
not enough, and frames the next technical step as a learned source-heldout
residual outcome policy. The proposed policy must remain target-GT-free at
decision time: target/test GT can be read only after outputs are written for
evaluation.

## Evidence From v338/v339

v338 tested the natural selector route: combine target-neighbor consistency
(TNC) scores with source-heldout KNN/local evidence and source-summary guards.
On focus6 (`stump, treehill, room, bicycle, bonsai, kitchen`), both the default
combined ranker and the rank1/cvar002 variant exactly matched v337diag:

| method | scenes | views | selected PSNR gain | selected SSIM gain | promotions |
|---|---:|---:|---:|---:|---:|
| v337diag | 6 | 170 | 0.301231403771 | 0.003460387180 | n/a |
| v338 default | 6 | 170 | 0.301231403771 | 0.003460387180 | 0 |
| v338b rank1/cvar002 | 6 | 170 | 0.301231403771 | 0.003460387180 | 0 |

The oracle headroom stayed unchanged at `+0.012506552`. This means the safe
TNC-plus-source-heldout ranker became a no-op. The post-hoc relaxation diagnostic
also did not rescue the route: the best target-GT-audited relaxed setting gave
only `+0.000240408751` dPSNR and `+0.000004195381` dSSIM, while creating 18
promotions, 10 bad promotions, and scene regressions. Because that relaxation
uses target GT after the fact, it is not a deployable selector.

v339 then tested the stronger manual-generator route. It added `tnc_reg`, a
target-GT-free TNC-regularized residual candidate, with source-summary admission
before downstream selection. The fair v339d full-stack focus6 result again
exactly matched v337diag:

| method | scenes | views | selected PSNR gain | selected SSIM gain | oracle headroom |
|---|---:|---:|---:|---:|---:|
| v337diag | 6 | 170 | 0.301231403771 | 0.003460387180 | +0.012506552 |
| v339d full stack | 6 | 170 | 0.301231403771 | 0.003460387180 | +0.012506552 |

Per-scene suppression explains the failure mode. `tnc_reg` was admitted only on
`room`; on `stump`, `treehill`, `bicycle`, `bonsai`, and `kitchen` it was
rejected by source-summary PSNR/SSIM deltas. On `room`, it was still weaker than
the already available adaptive/selected behavior. The learned-base full-room
probe was rejected with `source_summary_psnr_delta:-0.0135018206`.

The earlier v337 diagnostic gives the same warning in selector form: pure TNC
matched the strict oracle on only 37/170 focus6 views, and TNC-best was
`-0.051566646277` PSNR relative to the output on average. TNC has useful
observability value, but the raw signal is not aligned enough to be a selector
or a hand-written residual formula.

## Why Manual TNC Residual Is Insufficient

1. TNC measures target-neighbor self-consistency, not the residual outcome that
   matters for PSNR/SSIM/tail safety. It can prefer an internally consistent
   candidate that is worse against target GT.
2. Hand thresholds collapse into two bad regimes: strict settings are safe but
   no-op, while relaxed settings create small posterior gains with many bad
   promotions and scene regressions.
3. The current manual `tnc_reg` formula has too little context. It cannot learn
   when TNC agreement is trustworthy, when source support is out-of-distribution,
   or when residual shrink/blend/no-op is the right action.
4. Source-summary guards are doing their job by rejecting unsafe generated
   candidates, but that also proves the generated residual itself is not yet
   strong enough.
5. v311 already showed the broader risk: naive learned source-heldout risk can
   still suffer source-to-target proxy shift. The next learned policy must model
   outcome, uncertainty, and abstention explicitly, not merely replace a manual
   gate with another loose selector.

## Technical Motivation For v340

The next module should learn a source-heldout residual outcome policy. Instead
of encoding TNC as a direct rank threshold or residual step, train a small
target-blind policy on source-heldout views where the "target" image is a heldout
training view with known GT. For each heldout source view, generate the available
candidate residual outputs and label their outcomes:

- per-candidate PSNR/SSIM gain versus incumbent;
- local residual L1/MSE improvement;
- min-view/CVaR/tail risk;
- whether the candidate should be selected, blended, shrunk, or abstained.

The feature set should include source-heldout evidence, support geometry,
residual variance, source diversity, candidate deltas, source-summary margins,
OOD/reliability features, and TNC diagnostics. TNC should become an auxiliary
feature or loss term, not the primary decision rule.

At target/test apply time, the policy predicts residual outcome and confidence
without target GT, then chooses among:

- keep incumbent;
- select fixed/learned/hybrid/adaptive/generated candidate;
- blend or shrink a candidate when predicted upside is positive but uncertain;
- abstain when source-heldout evidence is out-of-distribution.

The policy must still pass existing safety layers: source-summary admission,
TNC contradiction/certificate checks where useful, and target-GT-free policy
trace logging before evaluation.

## Required Experimental Closure Checklist

1. Baseline freeze:
   - freeze focus6 comparators: v336c/v337/v338/v339d;
   - record exact roots, JSON summaries, and policy config for each comparator;
   - define v340 pass/fail before target results are inspected.

2. Source-heldout training dataset:
   - build leave-one-out or stride-heldout source views per scene;
   - generate incumbent and candidate outputs without using the heldout GT in
     features;
   - compute labels only after candidate outputs are fixed;
   - save per-view feature rows, outcome labels, and split provenance.

3. Model family and ablations:
   - start with a simple calibrated model before high-capacity networks;
   - compare source-only, TNC-only, source+TNC, and source+TNC+reliability;
   - include abstention/no-op as a first-class action;
   - report calibration error and predicted-vs-real gain scatter on source-heldout
     validation.

4. Safety gates:
   - require predicted PSNR and SSIM non-regression versus incumbent;
   - require tail/CVaR or min-view constraints before per-view switching;
   - keep source-summary admission as an outer guard;
   - log every rejected candidate with the failing margin.

5. Focus6 target-blind apply:
   - write all policy decisions before target/test GT is read;
   - evaluate macro PSNR/SSIM, per-scene gains, positive-view fraction, min-view
     gain, and oracle headroom;
   - compare directly against v339d, not only against fixed/base.

6. Promotion criteria:
   - must improve focus6 macro PSNR and not regress macro SSIM versus v339d;
   - must avoid scene-level PSNR/SSIM regressions against v339d;
   - must reduce oracle headroom or explain why remaining headroom is
     unreachable by the current candidate ladder;
   - must not rely on post-hoc target-GT threshold tuning.

7. Full9 only after focus6 pass:
   - run full9 with the frozen v340 policy and no new target tuning;
   - report scene table, tail table, qualitative panels, and failure cases;
   - keep rejected scenes as evidence instead of hiding them behind averages.

8. Reproducibility package:
   - save machine-readable summaries and Markdown reports;
   - include W&B offline run IDs or local run roots;
   - include policy traces showing no target/test GT in decision fields;
   - include an audit for source-heldout split leakage.

## Draft Verdict

v338 and v339 are valuable negative milestones. They rule out another manual
TNC rank threshold and a hand-crafted TNC residual as the main path to closure.
The next credible step is a learned source-heldout residual outcome policy that
predicts when residual evidence will actually improve heldout outcomes, uses TNC
as supporting evidence, and abstains when the source-to-target proxy is unsafe.
