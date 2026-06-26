# Render Delta MSE Diagnostic

- model_path: `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model`
- split: `test`
- base_method: `ours_26000_v104c_shrink_view_affine_min1_minviews1_flowers`
- candidate_method: `ours_26000_v106_podmoe_basepreserve_flowers`
- view_count: `22`
- mse_improved_views: `22`
- mse_worse_views: `0`
- mean_delta_mse: `-0.00000408`
- mean_cross_term_2ed: `-0.00000470`
- mean_delta_energy_d2: `0.00000062`
- mean_abs_delta: `0.00014529`

## Worst MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00007.png | -0.00000203 | -0.00000272 | 0.00000068 | 0.00016238 |
| 00005.png | -0.00000246 | -0.00000308 | 0.00000062 | 0.00014685 |
| 00016.png | -0.00000250 | -0.00000312 | 0.00000062 | 0.00014462 |
| 00020.png | -0.00000255 | -0.00000314 | 0.00000059 | 0.00014517 |
| 00006.png | -0.00000273 | -0.00000314 | 0.00000042 | 0.00010089 |

## Best MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00001.png | -0.00000624 | -0.00000689 | 0.00000065 | 0.00015512 |
| 00002.png | -0.00000592 | -0.00000654 | 0.00000062 | 0.00014635 |
| 00012.png | -0.00000587 | -0.00000659 | 0.00000072 | 0.00016504 |
| 00009.png | -0.00000569 | -0.00000636 | 0.00000067 | 0.00015768 |
| 00019.png | -0.00000529 | -0.00000597 | 0.00000069 | 0.00016090 |
