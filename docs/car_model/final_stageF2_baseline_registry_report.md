# Final Stage F2 Baseline Registry Report

Date: 2026-05-04

## Decision

`PASS`.

The final baseline registry now reproduces the R53.01 versus clean 22k table and flags R44.01 as render-losing against clean 22k. It also records missing or non-independent metric rows instead of letting them enter paper tables silently.

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/final_collect_baselines_and_results.py
```

## Outputs

```text
outputs/carnet/meshsplatopt/final_baseline_registry/final_results.json
outputs/carnet/meshsplatopt/final_baseline_registry/final_results.csv
outputs/carnet/meshsplatopt/final_baseline_registry/final_results.md
```

## Gate Checks

Actual integrity values:

```text
r53_vs_clean22k_reproduced: true
r44_flagged_render_losing_vs_clean22k: true
forbidden_long_method_vs_clean7k_headline: false
```

The key comparison reproduced by the collector is:

| candidate | baseline | pass | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepthMAE | dNormal | triangle reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R53.01 | parking.clean22k | true | +0.225748 | +0.013184 | -0.008421 | -0.002622 | -0.014647 | -0.847046 | 0.700000 |
| R44.01 | parking.clean22k | false | -1.310450 | -0.085909 | +0.094975 | +0.104890 | +1.050998 | -2.890186 | 0.908405 |

## Interpretation

The registry enforces the corrected final story:

- clean 22k is the parking render baseline for R53/R48/R55;
- clean 7k is historical only and cannot be the headline comparator for long method rows;
- R44 remains useful as topology/normal Pareto evidence but fails the render/depth baseline gate;
- R53 is the current headline parking row;
- R57/R60 negatives remain visible and motivate F3/F4 selector planning;
- R59 adds a render-positive/geometry-negative public Pareto row, which is useful evidence but not an all-metric pass.

Proceed to F3 cross-scene clean-to-compact feasibility planning.
