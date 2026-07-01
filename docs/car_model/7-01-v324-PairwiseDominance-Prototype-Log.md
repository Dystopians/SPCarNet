# v324 Pairwise Dominance Prototype Log

Date: 2026-07-01

## Status

v324 adds a new pairwise candidate-vs-incumbent dominance policy prototype to
`scripts/car_model/apply_source_heldout_support_transport_calibrator.py`.

This is a real train/eval pipeline interface, but it is not yet a successful
method. The prototype runs, writes diagnostics, and exposes the next bottleneck:
exact v322C incumbent replay and source-to-target pairwise transfer remain
fragile.

## Implemented Interface

New CLI switch:

```text
--enable_pairwise_dominance_policy
```

Main policy idea:

1. Run the existing v322C-style source reliability/KNN/gate stack first.
2. Treat that output as the target incumbent.
3. Evaluate every candidate against the incumbent using source-heldout pairwise
   features:
   candidate proxy, incumbent proxy, proxy delta, candidate/incumbent variant
   identity, and blend values.
4. Predict candidate-vs-incumbent deltas with ridge regression:
   objective, PSNR, SSIM, LPIPS, DISTS.
5. Require pairwise prediction, local KNN support, OOD, and source LOO global
   gates before overriding the incumbent.
6. If no candidate passes, output the incumbent unchanged.

New report payloads:

- `pairwise_dominance_policy`
- per-view `pairwise_dominance_predictions`
- per-view `pairwise_dominance_diagnostics`
- per-view `pairwise_dominance_decision`

## Validation So Far

Static checks:

```text
python -m py_compile scripts/car_model/apply_source_heldout_support_transport_calibrator.py
git diff --check -- scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

Smoke:

```text
outputs/carnet/spcarnet_v324_pairwise_smoke_treehill_20260701
```

The smoke run completed and wrote a full apply report. Pairwise dominance was
enabled, but strict gates accepted zero source LOO views:

```text
source_selected_counts = {'incumbent': 8, all candidates: 0}
source_accept_fraction = 0.0
```

Treehill relaxed focused probe:

```text
outputs/carnet/spcarnet_v324a_pairwise_relaxed_treehill_20260701
```

Result:

| method | selected PSNR gain | selected SSIM gain | safe vs fixed |
|---|---:|---:|---|
| v322C archived treehill | 0.103844 | 0.001665 | true |
| v324a relaxed treehill | 0.094703 | 0.001049 | true |

v324a improves over fixed but remains below archived v322C. It is not a valid
improvement.

## Key Finding

The main blocker is not the pairwise code path itself. The blocker is exact
incumbent replay.

The archived v322C treehill report had source reliability enabled with a small
safe set:

```text
source_selected_counts = {'fixed': 1, 'learned': 1, 'scene': 9}
source_mean_ssim_delta_vs_scene_selected = -9.38e-06
```

The current replay with the same obvious knobs can instead produce:

```text
source_selected_counts = {'fixed': 1, 'learned': 3, 'scene': 7}
source_mean_ssim_delta_vs_scene_selected = -1.26e-04
verdict = source reliability did not clear fixed-scene SSIM delta
```

Adding a predicted-SSIM gate enables source reliability but collapses it to an
almost fixed-only replay, also failing to match v322C.

This means any v324 comparison is unreliable until the exact archived v322C
incumbent replay is reproduced in the current code path.

## Relationship to Strict Oracle

The strict oracle diagnostic remains positive:

```text
docs/car_model/results/v323_strict_oracle_gap_summary.json
```

Full9 v322C strict oracle:

- improved views: `73 / 246`
- macro PSNR delta: `+0.011338`
- macro SSIM delta: `+0.000090`

So there is still candidate capacity. v324 is the correct conceptual direction,
but the current prototype has not learned a reliable decision boundary yet.

## Next Required Work

1. Build a dedicated archived-v322C replay audit that compares source LOO
   decisions from current code against the archived v322C reports.
2. Save target per-candidate proxies in reports so future diagnostics can
   analyze target-free features without rerunning full apply.
3. Rebuild pairwise source incumbents from exact replay, including KNN LOO
   decisions, not only source reliability LOO predictions.
4. Re-run v324 on focused scenes only after source replay matches archived v322C.
5. If pairwise source LOO still accepts zero views under strict gates, redesign
   the feature set or add calibrated pairwise LCB instead of loosening thresholds.

Final status: NOT COMPLETE.
