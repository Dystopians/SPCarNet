# Phase-J Integrated Runtime Profile

This summary aggregates raw per-scene runtime JSON files using only CPU-side JSON parsing.

## Summary

- label: `phasej_integrated_v2`
- scenes: `9`
- target views: `246`
- repeats per scene: `[2]`
- weighted ms/view: `951.410896`
- weighted FPS: `1.051071`
- weighted render ms/view: `37.090434`
- weighted adapter ms/view: `913.855245`
- weighted adapter/render ratio: `24.638570`
- max peak allocated MiB: `17703.596`
- max peak reserved MiB: `24498.000`
- render-only summary: `outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.json`
- render-only matched scenes: `9`
- weighted render-only compact ms/view: `35.179789`
- integrated/render-only compact ms ratio: `27.044247`
- adapter/render-only compact ms ratio: `25.976712`

## Per-Scene

| scene | views | repeats | ms/view | FPS | render ms | adapter ms | adapter/render | peak alloc MiB | covered | confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25 | 2 | 1116.839100 | 0.899391 | 31.260607 | 1085.184300 | 34.705190 | 10818.907 | 0.977215 | 3.402224 |
| bonsai | 37 | 2 | 791.108688 | 1.265503 | 36.910879 | 753.878722 | 20.421496 | 17053.017 | 0.976409 | 2.964140 |
| counter | 30 | 2 | 1134.619511 | 0.883180 | 45.699373 | 1088.201614 | 23.808257 | 17327.615 | 0.976259 | 2.956036 |
| flowers | 22 | 2 | 832.619785 | 1.201150 | 32.364652 | 799.868747 | 24.750449 | 10881.523 | 0.916098 | 1.972794 |
| garden | 24 | 2 | 849.438973 | 1.181303 | 35.903718 | 813.106209 | 22.633026 | 11966.776 | 0.976233 | 2.629218 |
| kitchen | 35 | 2 | 1010.527740 | 0.993914 | 42.970477 | 967.213790 | 22.502615 | 17167.559 | 0.986163 | 3.221464 |
| room | 39 | 2 | 1012.581573 | 0.989955 | 37.773754 | 974.120425 | 25.772492 | 17703.596 | 0.953533 | 2.732414 |
| stump | 16 | 2 | 872.583973 | 1.153822 | 31.794933 | 840.348024 | 26.384285 | 9935.544 | 0.939165 | 1.505427 |
| treehill | 18 | 2 | 849.546097 | 1.194664 | 30.359684 | 818.822917 | 26.898832 | 10301.594 | 0.704005 | 1.715288 |

## Render-Only Comparison

| scene | integrated ms/view | render-only compact ms/view | delta ms/view | integrated/render-only ratio | adapter/render-only ratio |
|---|---:|---:|---:|---:|---:|
| bicycle | 1116.839100 | 27.676855 | 1089.162245 | 40.352818 | 39.209090 |
| bonsai | 791.108688 | 36.306227 | 754.802461 | 21.789890 | 20.764447 |
| counter | 1134.619511 | 43.842166 | 1090.777344 | 25.879641 | 24.820891 |
| flowers | 832.619785 | 29.782082 | 802.837704 | 27.957072 | 26.857382 |
| garden | 849.438973 | 34.538410 | 814.900563 | 24.594038 | 23.542086 |
| kitchen | 1010.527740 | 41.507263 | 969.020477 | 24.345805 | 23.302278 |
| room | 1012.581573 | 35.583034 | 976.998539 | 28.456864 | 27.375980 |
| stump | 872.583973 | 28.792142 | 843.791831 | 30.306324 | 29.186714 |
| treehill | 849.546097 | 28.800940 | 820.745157 | 29.497165 | 28.430423 |

## Scope

- Aggregates existing profiler JSON files only; it does not render, call CUDA, compute image metrics, or import project GPU modules.
- Missing optional fields are emitted as `null` in JSON and blank cells in CSV.
