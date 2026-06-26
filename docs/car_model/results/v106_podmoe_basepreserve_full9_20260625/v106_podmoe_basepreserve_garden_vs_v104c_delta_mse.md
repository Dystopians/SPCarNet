# Render Delta MSE Diagnostic

- model_path: `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/garden/detached_model`
- split: `test`
- base_method: `ours_26000_v104c_shrink_view_affine_min1_minviews1_garden`
- candidate_method: `ours_26000_v106_podmoe_basepreserve_garden`
- view_count: `24`
- mse_improved_views: `23`
- mse_worse_views: `1`
- mean_delta_mse: `-0.00000157`
- mean_cross_term_2ed: `-0.00000197`
- mean_delta_energy_d2: `0.00000040`
- mean_abs_delta: `0.00008292`

## Worst MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00000.png | +0.00000116 | +0.00000060 | 0.00000055 | 0.00009983 |
| 00016.png | -0.00000072 | -0.00000102 | 0.00000030 | 0.00007229 |
| 00009.png | -0.00000098 | -0.00000136 | 0.00000037 | 0.00008875 |
| 00005.png | -0.00000115 | -0.00000152 | 0.00000038 | 0.00008782 |
| 00014.png | -0.00000117 | -0.00000150 | 0.00000033 | 0.00008095 |

## Best MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00004.png | -0.00000379 | -0.00000537 | 0.00000158 | 0.00011140 |
| 00012.png | -0.00000225 | -0.00000264 | 0.00000039 | 0.00009393 |
| 00001.png | -0.00000215 | -0.00000252 | 0.00000037 | 0.00007831 |
| 00008.png | -0.00000213 | -0.00000252 | 0.00000039 | 0.00009395 |
| 00007.png | -0.00000200 | -0.00000233 | 0.00000033 | 0.00008104 |
