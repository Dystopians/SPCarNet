# v113b OOT Tail-Safe Parent Gate Summary

Date: 2026-06-25

Protocol:

```text
candidate field: train/even
gate calibration: train/odd
final metrics: test
test GT use: final evaluation only
```

v113b adds two safety certificates to the render-realized parent gate:

1. Per-metric lower-tail certificate: the 5th-percentile calibration PSNR gain must be non-negative.
2. Out-of-trajectory support certificate: target/test camera centers with nonzero mask must remain inside the empirical train/odd-to-train/even support envelope. This uses camera poses and parent/candidate renders only, not target GT.

Metric triplets are `PSNR / SSIM / LPIPS`; higher PSNR/SSIM and lower LPIPS are better.

## Results

| scene | clean MeshSplatting | v106 parent | v110b prior strict gate | v113b OOT tail-safe gate | v113b vs clean | v113b vs v106 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| flowers | 19.682257 / 0.511822 / 0.394563 | 20.077723 / 0.531240 / 0.374393 | 20.077723 / 0.531240 / 0.374393 | 20.077723 / 0.531240 / 0.374393 | +0.395466 / +0.019418 / -0.020170 | +0.000000 / +0.000000 / +0.000000 | tail-safe fallback |
| garden | 25.029211 / 0.780035 / 0.201314 | 25.790945 / 0.799382 / 0.174480 | 25.430321 / 0.783703 / 0.186970 | 25.790945 / 0.799382 / 0.174480 | +0.761734 / +0.019347 / -0.026834 | +0.000000 / +0.000000 / +0.000000 | OOT scene fallback |

## Two-Scene Mean

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | 22.355734 | 0.645929 | 0.297939 |
| v106 parent | 22.934334 | 0.665311 | 0.274437 |
| v110b prior strict gate | 22.754022 | 0.657471 | 0.280681 |
| v113b OOT tail-safe gate | 22.934334 | 0.665311 | 0.274437 |
| v113b minus clean | +0.578600 | +0.019382 | -0.023502 |
| v113b minus v110b | +0.180312 | +0.007839 | -0.006245 |
| v113b minus v106 | +0.000000 | +0.000000 | +0.000000 |

## Gate Evidence

| scene | tail evidence | OOT evidence | final mask |
|---|---|---|---:|
| flowers | falls back before nonzero target edit because lower-tail PSNR safety is not proven | OOT pass is true, but irrelevant after tail fallback | 0.000000 |
| garden | nonzero candidate has a valid p05 dPSNR after threshold tightening | `mask_weighted_ood_fraction=0.090031 > 0.05`; target p95 center distance `0.806651` exceeds calibration p95 support `0.757181` | 0.000000 |

## Interpretation

v113b is a real method-level safety improvement over v110b: it prevents the observed garden held-out regression without using target GT. It restores the strict-gate branch to the v106 parent rather than letting a train/odd-positive but target-unsafe candidate overwrite the render.

This is still not a quality breakthrough over v106. It is a reliability milestone: the strict gate now has a principled no-regression fallback for the two completed representative scenes, while the next paper-level method must find a candidate that passes these certificates and improves beyond v106.

## Artifact Index

| artifact | path |
|---|---|
| flowers gate report | `docs/car_model/results/v113_oot_tail_20260625/flowers/flowers_v113b_oot_strict_gate_report.md` |
| flowers metrics | `docs/car_model/results/v113_oot_tail_20260625/flowers/flowers_ours_26000_v113b_oot_strict_parent_gate_flowers_test_results.json` |
| garden gate report | `docs/car_model/results/v113_oot_tail_20260625/garden/garden_v113b_oot_strict_gate_report.md` |
| garden metrics | `docs/car_model/results/v113_oot_tail_20260625/garden/garden_ours_26000_v113b_oot_strict_parent_gate_garden_test_results.json` |
