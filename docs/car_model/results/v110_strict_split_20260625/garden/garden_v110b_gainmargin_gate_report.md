# v109 Render-Realized Parent Gate Report

- parent_method: `ours_26000_v106_podmoe_basepreserve_garden`
- candidate_method: `ours_26000_v110_strict_train_even_candidate_garden`
- method_name: `ours_26000_v110b_strict_gainmargin_parent_gate_garden`
- calib_split: `train`
- target_split: `test`
- calib_view_subset: `odd`
- calib_candidate_count: `161`
- calib_selected_count: `64`
- no_target_gt_used_for_policy: `True`
- selected_policy: `{"dilate": 0, "frame_threshold": 0.0, "kernels": [1, 9, 25], "max_blend": 0.75, "softness": 0.002, "threshold": 0.004}`
- fallback_to_parent: `False`
- calib_score: `1.57675291`
- calib_mean_mask: `0.72910968`
- target_mean_mask: `0.69922511`
- target_views: `24`

## Selected Calibration Row

| dMSE | dPSNR | dSSIM | dLPIPS | p05 score gain | p95 delta MSE | pass |
|---:|---:|---:|---:|---:|---:|---:|
| -4.88663376e-04 | 0.79944015 | 0.03886564 | 0.00000000 | 0.05679893 | -2.13517342e-05 | yes |
