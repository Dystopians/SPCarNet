# Render Delta MSE Diagnostic

- model_path: `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/stump/detached_model`
- split: `test`
- base_method: `ours_26000_v104c_shrink_view_affine_min1_minviews1_stump`
- candidate_method: `ours_26000_v106_podmoe_basepreserve_stump`
- view_count: `16`
- mse_improved_views: `16`
- mse_worse_views: `0`
- mean_delta_mse: `-0.00000080`
- mean_cross_term_2ed: `-0.00000102`
- mean_delta_energy_d2: `0.00000022`
- mean_abs_delta: `0.00004813`

## Worst MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00010.png | -0.00000005 | -0.00000018 | 0.00000014 | 0.00003510 |
| 00015.png | -0.00000023 | -0.00000037 | 0.00000014 | 0.00003514 |
| 00012.png | -0.00000024 | -0.00000040 | 0.00000016 | 0.00003945 |
| 00004.png | -0.00000045 | -0.00000060 | 0.00000015 | 0.00003657 |
| 00005.png | -0.00000050 | -0.00000070 | 0.00000021 | 0.00005115 |

## Best MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00000.png | -0.00000363 | -0.00000386 | 0.00000023 | 0.00004751 |
| 00008.png | -0.00000121 | -0.00000153 | 0.00000032 | 0.00005928 |
| 00013.png | -0.00000094 | -0.00000111 | 0.00000017 | 0.00004365 |
| 00014.png | -0.00000088 | -0.00000110 | 0.00000022 | 0.00005475 |
| 00007.png | -0.00000077 | -0.00000100 | 0.00000024 | 0.00005998 |
