# Policy-Val Pruned Region Carriers

- input carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/carrier_unpruned.json`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/fit_evidence`
- output carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/carrier.json`
- input carriers: `64`
- output carriers: `45`
- candidate faces: `1061`
- atlas faces: `1050`
- retained faces: `226`
- removed faces: `824`
- prune unit: `face`
- input units: `1050`
- retained units: `226`
- prune alpha: `0.015625`
- greedy removals: `0`

## Retained Policy-Val Relative Gains

| view | samples | rel gain | sse before | sse gain |
|---|---:|---:|---:|---:|
| 00000 | 3335 | 0.00184475 | 74.26942444 | 0.13700848 |
| 00008 | 2734 | 0.00331024 | 81.35910797 | 0.26931827 |
| 00016 | 2000 | 0.00093668 | 84.83256531 | 0.07946124 |
| 00024 | 7183 | 0.00153580 | 88.24271393 | 0.13552272 |
| 00032 | 2165 | 0.00670285 | 63.05694962 | 0.42266119 |
| 00040 | 1153 | 0.00720548 | 34.89276123 | 0.25141906 |
| 00048 | 5594 | 0.00265658 | 81.93903351 | 0.21767735 |
| 00056 | 2071 | 0.00561282 | 36.83198166 | 0.20673128 |
| 00064 | 5208 | 0.00228283 | 84.99753571 | 0.19403508 |
| 00072 | 13129 | 0.00301884 | 319.58581543 | 0.96477822 |
| 00080 | 1357 | 0.00397605 | 34.33062744 | 0.13650025 |
| 00088 | 4178 | 0.00421628 | 157.78181458 | 0.66525168 |

This file is train-evidence only. The held-out test split is not used for
carrier pruning; final promotion still requires the downstream atlas apply
gate and image metrics.
