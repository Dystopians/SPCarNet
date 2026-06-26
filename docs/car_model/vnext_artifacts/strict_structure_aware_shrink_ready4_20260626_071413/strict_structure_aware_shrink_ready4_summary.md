# Strict Structure-Aware Shrink Ready4 Summary

日期：2026-06-26

场景：`counter,bonsai,room,garden`

策略：fixed `policy_val_structure_aware_shrink` with `risk_tau=0.002`, `l1_weight=1.0`, `gradient_weight=1.0`, strict no-target-GT apply.

## Result Table

| scene | artifact | protocol pass | accepted | alpha | changed fraction | PSNR | SSIM | LPIPS | delta PSNR vs parent | delta SSIM vs parent | delta LPIPS vs parent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | `counter_structure_shrink_tau002_20260626_0558` | true | true | 0.125 | 0.01234357 | 26.751171 | 0.862042 | 0.251955 | +0.00129890 | -0.00000906 | -0.00004268 |
| bonsai | `bonsai_structure_shrink_tau002_20260626_0718` | true | true | 0.25 | 0.00148974 | 28.865479 | 0.896003 | 0.259323 | +0.00113869 | -0.00000954 | -0.00001693 |
| room | `room_structure_shrink_tau002_20260626_0718` | true | true | 0.0625 | 0.00519912 | 28.739571 | 0.884797 | 0.249909 | +0.00046921 | +0.00000334 | -0.00001399 |
| garden | `garden_structure_shrink_tau002_20260626_071413` | true | true | 0.125 | 0.00205038 | 24.741142 | 0.754052 | 0.248015 | +0.00013924 | +0.00000316 | -0.00000791 |
| mean | - | 4/4 | 4/4 | - | - | - | - | - | +0.00076151 | -0.00000302 | -0.00002038 |

## Counts

- protocol pass: `4 / 4`
- target GT hidden from apply: `4 / 4`
- accepted nonzero: `4 / 4`
- PSNR better vs parent: `4 / 4`
- SSIM better vs parent: `2 / 4`
- LPIPS better vs parent: `4 / 4`

## Garden Delta Versus Previous Garden Face-SoftShrink

Garden structure-aware shrink vs old garden face-softshrink: `+0.00006294` PSNR / `+0.00000119` SSIM / `-0.00000468` LPIPS.

## Interpretation

Ready4 extends the strict structure-aware shrink table from counter/bonsai/room to garden. Garden is accepted/nonzero and improves PSNR/SSIM/LPIPS versus its Phase-F compact parent and the previous garden face-softshrink pilot. The result is still not full9 because five scenes lack evidence/carrier inputs.

This remains a milestone rather than a paper-final result: five full9 scenes still lack current evidence/carrier inputs, and the mean effect size is still tiny.
