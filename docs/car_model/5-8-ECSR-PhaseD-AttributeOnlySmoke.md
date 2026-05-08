# ECSR Phase-D Attribute-Only Recovery Smoke

This report covers the first executable Version-1 surface-attached
appearance recovery smoke after Phase-C static contraction. The runs
freeze topology and vertices, optimize only appearance attributes, sync
to W&B, render the held-out test split once, and evaluate sparse COLMAP
geometry. The held-out test metrics here are diagnostics; they were not
used to select a policy or tune a scene-specific setting.

| recovery | status | topology | extra tri red. | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM vs compact | dLPIPS vs compact |
|---|---|---|---|---|---|---|---|---|---|
| bicycle_C0001 | REJECT_SMOKE_REGRESSION | yes | 0.000024% | 23.1249 | 0.6366 | 0.3819 | -0.1686 | -0.0230 | +0.0496 |
| kitchen_C0019 | REJECT_SMOKE_REGRESSION | yes | 0.000021% | 27.7800 | 0.8737 | 0.2028 | -0.0386 | -0.0028 | +0.0036 |

Accepted by smoke rule: `0 / 2`

## Geometry Delta Vs Compact-Only

| recovery | dAbsRel | dDepthMAE | dNormalDeg |
|---|---|---|---|
| bicycle_C0001 | -0.000780 | -0.0106 | +0.0928 |
| kitchen_C0019 | -0.000000 | +0.0000 | +0.0272 |

## W&B Runs

| recovery | url |
|---|---|
| bicycle_C0001 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/m7kjav9k |
| kitchen_C0019 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ib90mh2e |

## Interpretation

The infrastructure is now real: materialized contraction checkpoints can
be loaded, topology can remain frozen through recovery, W&B logging works,
and RGB/geometry metrics are produced. However, this Version-1 smoke is
not accepted as a final method because it regresses held-out RGB metrics
relative to the compact-only source checkpoints. The next Phase-D step
must add policy-val controlled early stopping or a representation-attached
residual/delta mechanism instead of treating longer attribute fine-tuning
as automatically beneficial.
