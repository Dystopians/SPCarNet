# Final Stage SCE5 F82-vs-F95 Sparse Regression Diagnostic

Date: 2026-05-06

Decision: `PROCEED_TO_SCE6_WITH_DENSE_SENTINEL_FIX`

## Scope

This stage packages the current critical case: courtyard F82 fixed adaptive policy v5 is the accepted parent, and F95 render-geometry-anchor recovery is the strongest rejected candidate. F95 improves RGB and sparse-normal proxies, but fails the sparse-depth parent-Pareto gate.

## Artifacts

- Test analyzer: `outputs/carnet/meshsplatopt/final_stageSCE1_sparse_depth_regression/courtyard`
- Train sentinel cache, original resolution 4: `outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard/sentinel_cache.npz`
- Train sentinel cache, corrected resolution 8: `outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard_res8/sentinel_cache.npz`
- Train sentinel gate, original resolution 4: `outputs/carnet/meshsplatopt/final_stageSCE4_sentinel_gate/courtyard_f82_vs_f95`
- Train sentinel gate, corrected resolution 8: `outputs/carnet/meshsplatopt/final_stageSCE4_sentinel_gate/courtyard_res8_f82_vs_f95`

## Sparse Test Diagnosis

The SCE1 test analyzer reproduces the known F95 failure direction:

| row | AbsRel | Depth MAE | both-valid points | candidate-invalid | gate-critical |
| --- | ---: | ---: | ---: | ---: | ---: |
| F82 parent | 0.324888045 | 3.516864341 | 1961 | - | - |
| F95 candidate | 0.325786638 | 3.533150427 | 1961 | 8 | 98 |
| candidate-parent | +0.000898593 | +0.016286086 | - | +8 | +98 |

The failure is not a uniform collapse. It is concentrated in gate-critical correspondence clusters and in the far-depth quartile:

- far depth q4: AbsRel regresses by `+0.005750743`, MAE by `+0.071399680`;
- positive top-10% deltas: AbsRel regresses by `+0.245620606`, MAE by `+2.036418486`;
- boundary effects exist but are not the sole explanation: interior points also regress by AbsRel `+0.000730893`.

## Train Sentinel Predictiveness

The corrected resolution-8 train sentinel cache is training-safe:

- split: `train`
- no_test_leakage: `true`
- resolution: `8`
- views: `32`
- sentinels: `14167`
- F95-regressed candidate sentinels: `5394`

The corrected train sentinel gate predicts the same failure direction as the independent test sparse-depth analyzer:

| row | Sentinel AbsRel | Sentinel MAE | regressed sentinels | gate-critical |
| --- | ---: | ---: | ---: | ---: |
| F82 parent | 0.398396218 | 4.962074933 | - | - |
| F95 candidate | 0.400966688 | 4.997632539 | 5394 | 713 |
| candidate-parent | +0.002570470 | +0.035557606 | +5394 | +713 |

This confirms that train/calibration sentinels are useful as a pre-run diagnostic and training constraint. They are not a replacement for the final test split evaluation.

## Important Implementation Lesson

The first SCE2 cache was generated at `--resolution 4`, while the F95/SCE6 recovery path uses `--resolution 8`. That makes raw cached `px/py` unsafe unless the cache records its original render dimensions and the consumer rescales coordinates to the current rendered depth image.

The code now stores per-point `width` and `height` in the sentinel cache and rescales cached coordinates in:

- `utils/sparse_depth_parent_rollback.py`
- `scripts/car_model/meshsplatopt_sentinel_parent_pareto_gate.py`

The old resolution-4 cache remains a historical artifact only. New SCE6 evidence should use the corrected resolution-8 cache or a denser resolution-8 successor.

## Recommended SCE6 Configuration

The first valid rollback should use:

- cache: corrected train cache at resolution 8;
- loss space: `absrel`, because AbsRel is the parent-Pareto blocker and the top-delta failure is strongly relative-error driven;
- start: F95/F82 courtyard topology-frozen checkpoint lineage;
- rollback lambda: begin at `0.05`, then escalate only if the W&B pure rollback loss shows weak activation;
- margins: `0.0` for both AbsRel and MAE in the first diagnostic;
- max points per view: initially `500`, then increase with a denser cache if active points remain too low;
- keep render-normal anchor `0.01`, but treat teacher render and render-depth anchors as competing forces if sparse depth does not move.

The failure is localized and sentinel-predictive, so the correct decision is to proceed to SCE6. The main risk is insufficient sentinel density/weight rather than lack of a diagnostic signal.
