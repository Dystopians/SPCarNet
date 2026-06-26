# Strict Structure-Aware Shrink Multiscene Summary

日期：2026-06-26

场景：`counter,bonsai,room`

策略：fixed `policy_val_structure_aware_shrink` with `risk_tau=0.002`, `l1_weight=1.0`, `gradient_weight=1.0`, strict no-target-GT apply.

## Result Table

| scene | artifact | protocol pass | accepted | alpha | changed fraction | PSNR | SSIM | LPIPS | delta PSNR vs parent | delta SSIM vs parent | delta LPIPS vs parent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | `counter_structure_shrink_tau002_20260626_0558` | true | true | 0.125 | 0.01234357 | 26.751171 | 0.862042 | 0.251955 | +0.00129890 | -0.00000906 | -0.00004268 |
| bonsai | `bonsai_structure_shrink_tau002_20260626_0718` | true | true | 0.25 | 0.00148974 | 28.865479 | 0.896003 | 0.259323 | +0.00113869 | -0.00000954 | -0.00001693 |
| room | `room_structure_shrink_tau002_20260626_0718` | true | true | 0.0625 | 0.00519912 | 28.739571 | 0.884797 | 0.249909 | +0.00046921 | +0.00000334 | -0.00001399 |
| mean | - | 3/3 | 3/3 | - | - | - | - | - | +0.00096893 | -0.00000509 | -0.00002453 |

## Protocol Audit

All three runs report:

```text
selection_uses_test_gt=false
target_gt_visible_to_apply=false
target_forbidden_keys_stripped=true
target_apply_leak=false
thresholds_selected_on=train_policy_val
```

## Main Takeaway

Compared with the previous strict face-softshrink table, structure-aware shrink makes `room` accepted and nonzero instead of fallback/no-op. This is the most meaningful improvement in this batch.

The remaining limitation is effect size: mean gains are tiny and SSIM still slightly regresses on counter/bonsai. This supports a new milestone, not a final paper-grade claim.
