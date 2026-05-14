# Phase-S Coupled Render-Risk Selector Pilot

Date: 2026-05-13

This log records the first real Phase-S improvement after the top1/scale2
plateau. The new mechanism is a coupled face-set selector over the existing
train-only face-local residual plan.

## Motivation

The previous Phase-S top1/scale2 policy was safe but nearly inert:

- it materialized one face;
- it usually changed only three local vertices;
- most deltas were around `1e-6` to `1e-5`;
- visual differences required amplified diff maps;
- `garden` and `counter` could pass train-val while report-only test regressed.

The failure mode was not "no candidate faces." Most scenes have many certified
candidate faces. The real problem is that faces are not independent: several
good single-face residuals can interact badly under the real renderer and LPIPS
gate. Therefore the next method must select face sets, not just tune top1.

## Method

The new script is:

`scripts/car_model/ecsr_run_facelocal_coupled_selector.py`

It performs an outer-loop selector:

1. read the train-only face-local candidate plan;
2. build fixed candidate face sets;
3. materialize each set with the existing Phase-S facelocal operator;
4. run the existing Phase-K/S train-val render gate for each trial;
5. promote only trials that pass the inner gate and the outer selector threshold;
6. report held-out test metrics without using them for selection;
7. fallback to Phase-J when no trial is promoted.

The companion collector is:

`scripts/car_model/ecsr_collect_facelocal_coupled_selector_summary.py`

## Candidate Set Scoring

Two candidate-set modes were implemented:

- `topN`: preserve plan rank and take the first `N` faces.
- `scoreN`: sort by a train-only certificate score and take the top `N` faces.

The `scoreN` rule uses only fields already written in the candidate plan:

- `policy_val_proxy.relative_gain`
- `policy_val_proxy.samples`
- `validation_shrink.scale`
- `face_view_gain_certificate.beneficial_fraction`
- `face_view_gain_certificate.min_relative_gain`
- `face_view_consensus.consensus`
- `face_stats.consistency`
- `face_stats.pixel_count`
- `face_stats.view_hits`

The score is intentionally conservative:

```text
score_i =
  relative_gain_i
  * shrink_i
  * consensus_i
  * beneficial_fraction_i
  * (0.5 + 0.5 * min_view_gain_i)
  * consistency_i
  * log1p(samples_i)
  * log1p(pixel_count_i)
  * sqrt(view_hits_i)
```

No held-out test metric is used for scoring or promotion.

## Promotion Rule

The pilot uses the existing inner gate plus a stricter outer threshold:

```text
selector_min_trainval_balanced_delta = 0.00005
```

This threshold is important. Without it, many trials pass by being numerically
tiny. With it, Phase-S is only promoted when the train-val render evidence has
non-trivial amplitude. Otherwise the selected output falls back to Phase-J and
the effective test delta is zero.

## Pilot Commands

Representative command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  --scenes counter \
  --gpu 6 \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_counter \
  --trial_specs top1x2,score2x1,score4x1,score4x0.5,score8x0.5 \
  --wandb_group phase_s_facelocal_coupled_selector_v1_pilot_20260513 \
  --candidate_prefix facelocal_coupled_v1_pilot \
  --skip_failed_views \
  --selector_min_trainval_balanced_delta 0.00005
```

All training/evaluation stages that support W&B ran with online W&B logging.

## Five-Scene Pilot Summary

Collected summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_summary/summary.md`

| scene | candidates | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | reading |
|---|---:|---|---:|---:|---:|---:|---|
| bicycle | 7 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | multi-face trials failed train-val; top1 was too small to promote |
| flowers | 35 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | positive but below meaningful threshold |
| garden | 110 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | strict threshold prevents train-val false-positive from becoming a test regression |
| treehill | 84 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | multi-face trials regressed train-val |
| counter | 127 | score4/s1 | true | +0.000055313 | +0.000000417 | -0.000001699 | real coupled-selector win; old top1/s2 was negative |
| **mean** | - | - | 1/5 | **+0.000011063** | **+0.000000083** | **-0.000000340** | sparse but positive effective result |

## Eight Candidate-Scene Closure

After the first pilot, the same strict selector was run on the remaining
candidate-bearing scenes: `room`, `kitchen`, and `bonsai`. `stump` is excluded
from this table because its current Phase-S plan has zero candidates.

Collected summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_summary/summary_8candidate_scenes.md`

| scene | candidates | trials | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | best non-fallback trial |
|---|---:|---:|---|---:|---:|---:|---:|---|
| bicycle | 7 | 5 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | top1/s2 |
| flowers | 35 | 3 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | top1/s2 |
| garden | 110 | 5 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | score4/s1, but report-only test regresses |
| treehill | 84 | 3 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | top1/s2 |
| room | 76 | 3 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | top1/s2 |
| counter | 127 | 5 | score4/s1 | true | +0.000055313 | +0.000000417 | -0.000001699 | score4/s1 |
| kitchen | 145 | 3 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | score4/s0.5 |
| bonsai | 1266 | 3 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | top1/s2 failed train-val |
| **mean** | - | - | - | 1/8 | **+0.000006914** | **+0.000000052** | **-0.000000212** | sparse positive result |

The eight-scene result is cleaner than top1/s2 because the selector rejects
low-amplitude or train-val risky edits, but it is still sparse. It gives a real
counter improvement and avoids known garden/counter top1 regressions, but it
does not yet create a broad Phase-S margin.

## Counter: Why This Matters

The old top1/s2 result on counter was harmful:

```text
top1/s2 report-only test:
  dPSNR  = -0.000026703
  dSSIM  = -0.000000119
  dLPIPS = +0.000000060
```

The coupled selector finds a four-face score-ranked set:

```text
score4/s1 report-only test:
  dPSNR  = +0.000055313
  dSSIM  = +0.000000417
  dLPIPS = -0.000001699
```

This is still small, but it is a real directional fix: the method changes the
old negative top1 outcome into a positive all-metric outcome under the same
report-only test protocol.

## Qualitative Evidence

Counter qualitative assets:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_qualitative/counter_qualitative_summary.md`

Panels show Phase-J, old top1/s2, coupled score4/s1, GT, and amplified
`|coupled - Phase-J|`.

![counter view 00002](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_qualitative/counter_00002_phasej_top1_coupled_x120diff.png)

![counter view 00026](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_qualitative/counter_00026_phasej_top1_coupled_x120diff.png)

Per-view diagnostic examples:

| view | top1 dPSNR | coupled dPSNR | coupled-vs-Phase-J MAE | max abs |
|---|---:|---:|---:|---:|
| 00002 | -0.000003491 | +0.000518573 | 0.000310 | 14 |
| 00026 | -0.000577692 | +0.000369517 | 0.001723 | 20 |
| 00000 | +0.000002077 | +0.000289366 | 0.001078 | 10 |

The full-frame visual change is still subtle, but the counter case finally has
a clear local diagnostic: the old top1 edit hurts selected views, while the
coupled score4 edit improves them.

## Honest Assessment

This is a meaningful method upgrade, but not a solved final paper result.

Positive:

- real train/eval pipeline change, not a README-only claim;
- uses fixed train-only scoring and train-val-only promotion;
- includes a fallback rule that prevents tiny or risky edits from hurting the
  effective method output;
- fixes counter, where top1/s2 was negative.

Still weak:

- only `1/5` pilot scenes promote under the meaningful threshold;
- mean gain is positive but still small;
- visual improvement remains subtle;
- `stump` has zero candidate faces and requires candidate discovery work;
- the selector does not yet model explicit pairwise view/geometry overlap.

## Next Step

The next step is not another scalar threshold sweep. The current strict selector
has established a useful safety rule and one positive scene, but the low
acceptance rate shows that score-only face sets are still too weak. The next
real method increment should add explicit pairwise/coupled risk terms:

- view-support overlap penalty;
- geometry-neighborhood overlap penalty;
- residual-direction conflict penalty;
- per-view tail-risk or CVaR on train-policy-val residual prediction;
- a stump-specific candidate discovery relaxation, because current strict
  certificates produce zero candidates there.

Until those are implemented and validated, Phase-J remains the main strong
paper-facing result, while coupled Phase-S is a promising but sparse repair
branch.
