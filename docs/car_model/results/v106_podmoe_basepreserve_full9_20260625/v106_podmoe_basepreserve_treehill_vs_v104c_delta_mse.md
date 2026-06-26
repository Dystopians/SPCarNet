# Render Delta MSE Diagnostic

- model_path: `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/treehill/detached_model`
- split: `test`
- base_method: `ours_26000_v104c_shrink_view_affine_min1_minviews1_treehill`
- candidate_method: `ours_26000_v106_podmoe_basepreserve_treehill`
- view_count: `18`
- mse_improved_views: `16`
- mse_worse_views: `2`
- mean_delta_mse: `-0.00000212`
- mean_cross_term_2ed: `-0.00000253`
- mean_delta_energy_d2: `0.00000041`
- mean_abs_delta: `0.00009780`

## Worst MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00000.png | +0.00000053 | +0.00000026 | 0.00000027 | 0.00006380 |
| 00009.png | +0.00000052 | +0.00000005 | 0.00000047 | 0.00011328 |
| 00005.png | -0.00000075 | -0.00000116 | 0.00000040 | 0.00009532 |
| 00008.png | -0.00000098 | -0.00000148 | 0.00000050 | 0.00011687 |
| 00007.png | -0.00000127 | -0.00000157 | 0.00000030 | 0.00007417 |

## Best MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00010.png | -0.00000714 | -0.00000766 | 0.00000052 | 0.00012463 |
| 00004.png | -0.00000432 | -0.00000483 | 0.00000050 | 0.00011344 |
| 00011.png | -0.00000378 | -0.00000430 | 0.00000052 | 0.00012705 |
| 00003.png | -0.00000299 | -0.00000322 | 0.00000024 | 0.00005493 |
| 00001.png | -0.00000255 | -0.00000286 | 0.00000031 | 0.00007143 |
