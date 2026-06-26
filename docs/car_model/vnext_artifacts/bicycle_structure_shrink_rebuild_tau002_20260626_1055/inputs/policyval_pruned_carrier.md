# Policy-Val Pruned Region Carriers

- input carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/carrier_unpruned.json`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/fit_evidence`
- output carrier json: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/carrier.json`
- input carriers: `64`
- output carriers: `60`
- candidate faces: `3080`
- atlas faces: `3055`
- retained faces: `1256`
- removed faces: `1799`
- prune unit: `face`
- input units: `3055`
- retained units: `1256`
- prune alpha: `0.015625`
- greedy removals: `0`

## Retained Policy-Val Relative Gains

| view | samples | rel gain | sse before | sse gain |
|---|---:|---:|---:|---:|
| 00000 | 9393 | 0.01481839 | 196.75878906 | 2.91564808 |
| 00008 | 2301 | 0.00888897 | 55.60684204 | 0.49428758 |
| 00016 | 6727 | 0.01154608 | 165.50041199 | 1.91088110 |
| 00024 | 2408 | 0.00532827 | 45.03221893 | 0.23994379 |
| 00032 | 10780 | 0.00763784 | 403.22766113 | 3.07978884 |
| 00040 | 5401 | 0.01322368 | 69.80381012 | 0.92306322 |
| 00048 | 5546 | 0.00965413 | 133.24261475 | 1.28634086 |
| 00056 | 3124 | 0.00473739 | 82.35544586 | 0.39014987 |
| 00064 | 651 | 0.00925803 | 2.17344999 | 0.02012186 |
| 00072 | 6790 | 0.00395823 | 121.58057404 | 0.48124421 |
| 00080 | 52 | 0.00849001 | 0.30539501 | 0.00259281 |
| 00088 | 7688 | 0.00567163 | 189.97418213 | 1.07746273 |

This file is train-evidence only. The held-out test split is not used for
carrier pruning; final promotion still requires the downstream atlas apply
gate and image metrics.
