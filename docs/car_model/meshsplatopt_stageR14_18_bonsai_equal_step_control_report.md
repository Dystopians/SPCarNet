# MeshSplatOpt Stage R14.18 Bonsai Equal-Step Control Report

Date: 2026-05-02

## Decision

`CONTROL_PASS_NEGATIVE_FOR_SNAP_GAIN`.

This stage runs a W&B-logged 200-step baseline continuation on `bonsai`: the unedited 2000iter sparse-depth baseline is copied, resumed from iteration 2000, trained to iteration 2200, rendered, evaluated, and checked with the sparse COLMAP geometry proxy.

This is the first equal-step control for the R14.17 non-delete snap recovery diagnostic.

## W&B

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kic0euiq
```

## Equal-Step Comparison

| row | edit | iteration | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline continuation | none | `2200` | `13.274771690368652` | `0.2403060346841812` | `0.6113919019699097` | `0.47338970412280024` | `4.765895956720541` | `49.19677426124215` |
| snap recovery | `SNAP_VERTICES` | `2200` | `13.273988723754883` | `0.24039088189601898` | `0.6116319894790649` | `0.47445281696526337` | `4.772623802825101` | `49.315686202793366` |

Snap minus baseline continuation:

| PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---:|---:|---:|---:|---:|---:|
| `-0.0007829666137695312` | `0.00008484721183776855` | `0.00024008750915527344` | `0.0010631128424631347` | `0.006727846104559882` | `0.11891194155121613` |

## Interpretation

The equal-step control is negative for a performance-gain claim: baseline continuation is slightly better on PSNR, LPIPS, AbsRel, Depth MAE, and normal angle, while snap recovery is only slightly better on SSIM.

This does not invalidate the R14.14-R14.17 safety and stability evidence. It does mean the current checkpoint-statistics snap selector should be reported as a safe real-edit mechanism, not as a demonstrated quality-improving method.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_18_bonsai_baseline_continuation_200step/teacher_recovery_run_report.json`
- `outputs/carnet/meshsplatopt/stageR14_18_bonsai_baseline_continuation_200step/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_18_bonsai_baseline_continuation_200step/recovery_model/geometry_eval_colmap/iter_2200_max500.json`
