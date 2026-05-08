# ECSR Phase-D Constrained Attribute Recovery

This report collects topology-frozen representation-level recovery runs.
The checkpoint topology is fixed, the rendered images are not edited, and
W&B is enabled during training. A run is accepted only if the recovered
checkpoint is topology-stable and does not regress PSNR, SSIM, or LPIPS
against compact-only. Compact-ELA is reported as the image-space teacher
or upper bound, not as the accepted representation-level method.

| scene | status | topology | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM vs compact | dLPIPS vs compact | dPSNR vs ELA | dSSIM vs ELA | dLPIPS vs ELA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flowers | REJECT_RGB_REGRESSION | yes | 20.1200 | 0.6043 | 0.3765 | -0.0459 | -0.0023 | +0.0024 | n/a | n/a | n/a |
| garden | REJECT_RGB_REGRESSION | yes | 25.7909 | 0.7835 | 0.2377 | -0.0543 | -0.0020 | +0.0022 | n/a | n/a | n/a |

Accepted by strict diagnostic rule: `0 / 2`.

## Geometry Delta Vs Compact-Only

| scene | dAbsRel | dDepthMAE | dNormalDeg |
|---|---|---|---|
| flowers | +0.003883 | +0.1185 | +1.5412 |
| garden | -0.002011 | -0.0405 | +0.4729 |

## W&B Runs

| scene | url |
|---|---|
| flowers | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/tlidfu7x |
| garden | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/bhi8cxgl |

## Interpretation

This is a strict Phase-D diagnostic, not a headline method unless it
passes the table above. Negative rows are useful because they separate
representation-level recovery failures from image-space ELA gains and
prevent us from promoting a method that only looks good after test-time
post-render correction.
