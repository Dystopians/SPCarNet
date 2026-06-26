# Policy-Val Pruned Region Carriers

- input carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/carrier_unpruned.json`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/fit_evidence`
- output carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/carrier.json`
- input carriers: `64`
- output carriers: `57`
- candidate faces: `2711`
- atlas faces: `2699`
- retained faces: `1315`
- removed faces: `1384`
- prune unit: `face`
- input units: `2699`
- retained units: `1315`
- prune alpha: `0.015625`
- greedy removals: `0`

## Retained Policy-Val Relative Gains

| view | samples | rel gain | sse before | sse gain |
|---|---:|---:|---:|---:|
| 00000 | 12107 | 0.00211622 | 80.04373169 | 0.16939037 |
| 00008 | 20232 | 0.00259127 | 304.87847900 | 0.79002150 |
| 00016 | 15090 | 0.00091275 | 239.89065552 | 0.21896013 |
| 00024 | 18700 | 0.00158486 | 214.12030029 | 0.33935106 |
| 00032 | 16385 | 0.00158127 | 215.07162476 | 0.34008672 |
| 00040 | 13471 | 0.00436967 | 30.95227623 | 0.13525137 |
| 00048 | 23632 | 0.00021778 | 112.78732300 | 0.02456255 |
| 00056 | 12352 | 0.00108727 | 238.97421265 | 0.25983014 |
| 00064 | 8200 | 0.00400268 | 53.63580322 | 0.21468684 |
| 00072 | 9682 | 0.00262293 | 104.97964478 | 0.27535472 |
| 00080 | 14555 | 0.00217451 | 509.69729614 | 1.10833987 |
| 00088 | 15280 | 0.00386168 | 210.44201660 | 0.81265904 |

This file is train-evidence only. The held-out test split is not used for
carrier pruning; final promotion still requires the downstream atlas apply
gate and image metrics.
