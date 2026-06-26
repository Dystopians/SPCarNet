# Render Delta MSE Diagnostic

- model_path: `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/room/detached_model`
- split: `test`
- base_method: `ours_26000_v104c_shrink_view_affine_min1_minviews1_room`
- candidate_method: `ours_26000_v106_podmoe_basepreserve_room`
- view_count: `39`
- mse_improved_views: `38`
- mse_worse_views: `1`
- mean_delta_mse: `-0.00000059`
- mean_cross_term_2ed: `-0.00000087`
- mean_delta_energy_d2: `0.00000028`
- mean_abs_delta: `0.00004623`

## Worst MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00036.png | +0.00000042 | +0.00000030 | 0.00000012 | 0.00002915 |
| 00030.png | -0.00000009 | -0.00000025 | 0.00000016 | 0.00003976 |
| 00027.png | -0.00000016 | -0.00000030 | 0.00000014 | 0.00003575 |
| 00031.png | -0.00000023 | -0.00000033 | 0.00000010 | 0.00002469 |
| 00007.png | -0.00000029 | -0.00000050 | 0.00000021 | 0.00004996 |

## Best MSE Views

| image | delta MSE | 2ed | d2 | mean abs delta |
|---|---:|---:|---:|---:|
| 00029.png | -0.00000225 | -0.00000259 | 0.00000033 | 0.00007995 |
| 00023.png | -0.00000144 | -0.00000161 | 0.00000016 | 0.00004023 |
| 00019.png | -0.00000123 | -0.00000144 | 0.00000022 | 0.00005207 |
| 00033.png | -0.00000095 | -0.00000122 | 0.00000027 | 0.00004198 |
| 00002.png | -0.00000094 | -0.00000114 | 0.00000020 | 0.00005018 |
