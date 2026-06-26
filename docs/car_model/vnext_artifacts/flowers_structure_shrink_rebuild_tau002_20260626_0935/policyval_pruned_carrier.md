# Policy-Val Pruned Region Carriers

- input carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier_unpruned.json`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/fit_evidence`
- output carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json`
- input carriers: `64`
- output carriers: `57`
- candidate faces: `931`
- atlas faces: `930`
- retained faces: `342`
- removed faces: `588`
- prune unit: `face`
- input units: `930`
- retained units: `342`
- prune alpha: `0.015625`
- greedy removals: `0`

## Retained Policy-Val Relative Gains

| view | samples | rel gain | sse before | sse gain |
|---|---:|---:|---:|---:|
| 00000 | 3351 | 0.00566918 | 75.91434479 | 0.43037234 |
| 00008 | 1219 | 0.00063916 | 12.89365101 | 0.00824112 |
| 00016 | 1978 | 0.00131169 | 57.27767181 | 0.07513069 |
| 00024 | 1243 | 0.00267069 | 43.91946411 | 0.11729535 |
| 00032 | 2367 | 0.00115048 | 53.43304062 | 0.06147356 |
| 00040 | 1428 | 0.00021797 | 27.81882477 | 0.00606367 |
| 00048 | 5162 | 0.00673242 | 138.27197266 | 0.93090469 |
| 00056 | 2189 | 0.00073438 | 22.96689224 | 0.01686651 |
| 00064 | 4795 | 0.00306990 | 137.47758484 | 0.42204185 |
| 00072 | 5443 | 0.00449253 | 83.34004211 | 0.37440773 |
| 00080 | 1578 | 0.00212930 | 6.67782593 | 0.01421909 |
| 00088 | 3209 | 0.00610461 | 56.63948059 | 0.34576193 |

This file is train-evidence only. The held-out test split is not used for
carrier pruning; final promotion still requires the downstream atlas apply
gate and image metrics.
