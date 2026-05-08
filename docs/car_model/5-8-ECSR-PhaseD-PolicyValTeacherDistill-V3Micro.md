# ECSR Phase-D Constrained Attribute Recovery

This report collects topology-frozen representation-level recovery runs.
The checkpoint topology is fixed, the rendered images are not edited, and
W&B is enabled during training. A run is accepted only if the recovered
checkpoint is topology-stable and does not regress PSNR, SSIM, or LPIPS
against compact-only. Compact-ELA is reported as the image-space teacher
or upper bound, not as the accepted representation-level method.

| scene | status | topology | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM vs compact | dLPIPS vs compact | dPSNR vs ELA | dSSIM vs ELA | dLPIPS vs ELA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flowers | REJECT_RGB_REGRESSION | yes | 20.1614 | 0.6064 | 0.3743 | -0.0045 | -0.0002 | +0.0002 | n/a | n/a | n/a |
| garden | REJECT_RGB_REGRESSION | yes | 25.8398 | 0.7853 | 0.2357 | -0.0054 | -0.0002 | +0.0002 | n/a | n/a | n/a |

Accepted by strict diagnostic rule: `0 / 2`.

## Geometry Delta Vs Compact-Only

| scene | dAbsRel | dDepthMAE | dNormalDeg |
|---|---|---|---|
| flowers | +0.003860 | +0.1177 | +1.5203 |
| garden | -0.002011 | -0.0405 | +0.4255 |

## W&B Runs

| scene | url |
|---|---|
| flowers | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/9n5pda94 |
| garden | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/nhami59o |

## Interpretation

This is a strict Phase-D diagnostic, not a headline method unless it
passes the table above. Negative rows are useful because they separate
representation-level recovery failures from image-space ELA gains and
prevent us from promoting a method that only looks good after test-time
post-render correction.
