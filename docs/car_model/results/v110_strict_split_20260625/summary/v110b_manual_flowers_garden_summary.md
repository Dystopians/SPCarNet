# v110b Manual Strict-Gate Summary: Flowers and Garden

Date: 2026-06-25

This file records the manual strict-gate follow-up after the v110 runner argument bug was fixed. The protocol is still strict for the candidate/gate stage:

```text
candidate field: train/even
gate calibration: train/odd
final metrics: test
test GT use: final evaluation only
```

The comparator named `clean MeshSplatting` is the local paper-reproduction baseline:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/<scene>/results.json
```

Metric triplets are `PSNR / SSIM / LPIPS`; higher PSNR/SSIM and lower LPIPS are better.

## Scene Results

| scene | clean MeshSplatting | v106 parent | v110b gain-margin gate | v110b vs clean | v110b vs v106 | gate decision |
|---|---:|---:|---:|---:|---:|---|
| flowers | 19.682257 / 0.511822 / 0.394563 | 20.077723 / 0.531240 / 0.374393 | 20.077723 / 0.531240 / 0.374393 | +0.395466 / +0.019418 / -0.020170 | +0.000000 / +0.000000 / +0.000000 | fallback to parent |
| garden | 25.029211 / 0.780035 / 0.201314 | 25.790945 / 0.799382 / 0.174480 | 25.430321 / 0.783703 / 0.186970 | +0.401110 / +0.003668 / -0.014345 | -0.360624 / -0.015679 / +0.012489 | accepted nonzero mask |

## Two-Scene Mean

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | 22.355734 | 0.645929 | 0.297939 |
| v106 parent | 22.934334 | 0.665311 | 0.274437 |
| v110b gain-margin gate | 22.754022 | 0.657471 | 0.280681 |
| v110b minus clean | +0.398288 | +0.011543 | -0.017257 |
| v110b minus v106 | -0.180312 | -0.007839 | +0.006245 |

## Interpretation

`v110b` is safer than the default v110 flowers gate because it blocks the flowers false accept and preserves v106 exactly. However, garden still shows a held-out test regression relative to the v106 parent after train/odd calibration selected a nonzero mask. Therefore v110b is a useful diagnosis and safety patch, not a promoted quality method.

Current paper-safe status:

- `v106 POD-MoE base-preserve` remains the best verified quality line in this package.
- `v110/v110b` demonstrates that strict train/odd gating is necessary but not yet sufficient for out-of-trajectory safety.
- The next method step should model cross-view/trajectory risk directly instead of trusting a single train/odd gate score.

## Artifact Index

| artifact | path |
|---|---|
| flowers v110b gate report | `docs/car_model/results/v110_strict_split_20260625/flowers/flowers_v110b_gainmargin_gate_report.md` |
| flowers v110b metrics | `docs/car_model/results/v110_strict_split_20260625/flowers/flowers_ours_26000_v110b_strict_gainmargin_parent_gate_flowers_test_results.json` |
| garden v110b gate report | `docs/car_model/results/v110_strict_split_20260625/garden/garden_v110b_gainmargin_gate_report.md` |
| garden v110b metrics | `docs/car_model/results/v110_strict_split_20260625/garden/garden_ours_26000_v110b_strict_gainmargin_parent_gate_garden_test_results.json` |
