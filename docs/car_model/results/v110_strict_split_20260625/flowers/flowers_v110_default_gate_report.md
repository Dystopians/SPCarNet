# v109 Render-Realized Parent Gate Report

- parent_method: `ours_26000_v106_podmoe_basepreserve_flowers`
- candidate_method: `ours_26000_v110_strict_train_even_candidate_flowers`
- method_name: `ours_26000_v110_strict_train_even_odd_parent_gate_flowers`
- calib_split: `train`
- target_split: `test`
- calib_view_subset: `odd`
- calib_candidate_count: `151`
- calib_selected_count: `64`
- no_target_gt_used_for_policy: `True`
- selected_policy: `{"dilate": 0, "frame_threshold": 0.0, "kernels": [1, 9, 25], "max_blend": 0.5, "softness": 0.0, "threshold": 0.001}`
- fallback_to_parent: `False`
- calib_score: `0.88661005`
- calib_mean_mask: `0.49640538`
- target_mean_mask: `0.49308174`
- target_views: `22`

## Selected Calibration Row

| dMSE | dPSNR | dSSIM | dLPIPS | p05 score gain | p95 delta MSE | pass |
|---:|---:|---:|---:|---:|---:|---:|
| -7.83816147e-04 | 0.38228006 | 0.02521650 | 0.00000000 | 0.08158028 | -1.83222815e-04 | yes |
