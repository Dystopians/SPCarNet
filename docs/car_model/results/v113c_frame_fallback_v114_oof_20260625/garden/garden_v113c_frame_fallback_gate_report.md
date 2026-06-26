# v109 Render-Realized Parent Gate Report

- parent_method: `ours_26000_v106_podmoe_basepreserve_garden`
- candidate_method: `ours_26000_v110_strict_train_even_candidate_garden`
- method_name: `ours_26000_v113c_oot_frame_parent_gate_garden`
- calib_split: `train`
- target_split: `test`
- calib_view_subset: `odd`
- calib_candidate_count: `161`
- calib_selected_count: `64`
- no_target_gt_used_for_policy: `True`
- selected_policy: `{"dilate": 0, "frame_threshold": 0.0, "kernels": [1, 9, 25], "max_blend": 0.75, "softness": 0.002, "threshold": 0.014}`
- fallback_to_parent: `False`
- calib_score: `1.54367467`
- calib_mean_mask: `0.50240009`
- target_mean_mask: `0.30709477`
- target_views: `24`
- oot_gate_mode: `frame_fallback`
- oot_gate_pass: `False`
- oot_fallback_reason: `mask_weighted_fraction_exceeds_support`

## Selected Calibration Row

| dMSE | dPSNR | dSSIM | dLPIPS | p05 score gain | p05 dPSNR | p05 dSSIM | p05 dLPIPS | p95 delta MSE | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -4.83453210e-04 | 0.78916886 | 0.03772529 | 0.00000000 | 0.11526084 | 0.00408983 | 0.00849563 | 0.00000000 | -2.83918343e-05 | yes |

## Out-of-Trajectory Gate

- source_fit_view_count: `81`
- calib_p95_center_dist: `0.75718139`
- target_p95_center_dist: `0.80665142`
- center_dist_threshold: `0.75718139`
- target_frame_fraction: `0.08333333`
- mask_weighted_ood_fraction: `0.09003060`
- applied_scene_fallback: `False`
- applied_frame_fallback: `True`
- frame_fallback_count: `2`
- mask_weighted_fraction_after_frame_fallback: `0.00000000`
