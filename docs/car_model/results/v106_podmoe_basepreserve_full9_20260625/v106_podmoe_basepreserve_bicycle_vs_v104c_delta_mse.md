# Render Delta MSE Diagnostic

- model_path: `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/bicycle/detached_model`
- split: `test`
- base_method: `ours_26000_v104c_shrink_view_affine_min1_minviews1_bicycle`
- candidate_method: `ours_26000_v106_podmoe_basepreserve_bicycle`
- view_count: `25`
- mse_improved_views: `25`
- mse_worse_views: `0`
- mean_delta_mse: `-0.00000149`
- mean_cross_term_2ed: `-0.00000179`
- mean_delta_energy_d2: `0.00000029`
- mean_abs_delta: `0.00006962`

## Worst MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00016.png | -0.00000051 | -0.00000081 | 0.00000029 | 0.00006850 |
| 00022.png | -0.00000056 | -0.00000075 | 0.00000019 | 0.00004814 |
| 00002.png | -0.00000066 | -0.00000103 | 0.00000036 | 0.00007196 |
| 00024.png | -0.00000080 | -0.00000098 | 0.00000017 | 0.00004305 |
| 00013.png | -0.00000095 | -0.00000119 | 0.00000024 | 0.00005785 |

## Best MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00004.png | -0.00000434 | -0.00000479 | 0.00000046 | 0.00011047 |
| 00011.png | -0.00000306 | -0.00000356 | 0.00000051 | 0.00011394 |
| 00000.png | -0.00000305 | -0.00000335 | 0.00000030 | 0.00006605 |
| 00001.png | -0.00000197 | -0.00000227 | 0.00000030 | 0.00006973 |
| 00010.png | -0.00000173 | -0.00000214 | 0.00000041 | 0.00008939 |
