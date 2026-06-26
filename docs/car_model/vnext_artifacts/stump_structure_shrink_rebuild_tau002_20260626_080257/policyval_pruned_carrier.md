# Policy-Val Pruned Region Carriers

- input carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/carrier_unpruned.json`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/fit_evidence`
- output carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/carrier.json`
- input carriers: `64`
- output carriers: `30`
- candidate faces: `533`
- atlas faces: `519`
- retained faces: `85`
- removed faces: `434`
- prune unit: `face`
- input units: `519`
- retained units: `85`
- prune alpha: `0.015625`
- greedy removals: `3`

## Retained Policy-Val Relative Gains

| view | samples | rel gain | sse before | sse gain |
|---|---:|---:|---:|---:|
| 00000 | 2446 | 0.00624835 | 22.87654495 | 0.14294058 |
| 00008 | 1498 | 0.00078969 | 16.24843597 | 0.01283126 |
| 00016 | 2797 | 0.00342571 | 46.59388351 | 0.15961733 |
| 00024 | 620 | 0.00200459 | 5.10933447 | 0.01024212 |
| 00032 | 3080 | 0.00108377 | 26.52594948 | 0.02874800 |
| 00040 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| 00048 | 2173 | 0.00187677 | 21.79256439 | 0.04089955 |
| 00056 | 2975 | 0.00070745 | 26.19834709 | 0.01853415 |
| 00064 | 172 | 0.00000000 | 2.71389151 | 0.00000000 |
| 00072 | 588 | 0.00012231 | 1.55068350 | 0.00018966 |
| 00080 | 31 | 0.00000000 | 0.04888678 | 0.00000000 |
| 00088 | 126 | 0.00802940 | 0.50676996 | 0.00406906 |

This file is train-evidence only. The held-out test split is not used for
carrier pruning; final promotion still requires the downstream atlas apply
gate and image metrics.
