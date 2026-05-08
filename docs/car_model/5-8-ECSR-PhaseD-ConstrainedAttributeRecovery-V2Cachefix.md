# ECSR Phase-D Constrained Attribute Recovery

This report collects topology-frozen representation-level recovery runs.
The checkpoint topology is fixed, the rendered images are not edited, and
W&B is enabled during training. A run is accepted only if the recovered
checkpoint is topology-stable and does not regress PSNR, SSIM, or LPIPS
against compact-only. Compact-ELA is reported as the image-space teacher
or upper bound, not as the accepted representation-level method.

| scene | status | topology | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM vs compact | dLPIPS vs compact | dPSNR vs ELA | dSSIM vs ELA | dLPIPS vs ELA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | REJECT_RGB_REGRESSION | yes | 23.0213 | 0.6329 | 0.3859 | -0.2722 | -0.0268 | +0.0536 | -0.8914 | -0.0609 | +0.1056 |
| flowers | REJECT_RGB_REGRESSION | yes | 19.4622 | 0.4875 | 0.4279 | -0.2097 | -0.0242 | +0.0331 | -0.7206 | -0.0598 | +0.0769 |
| treehill | REJECT_RGB_REGRESSION | yes | 20.7888 | 0.5556 | 0.4486 | -0.1352 | -0.0086 | +0.0425 | -0.4096 | -0.0326 | +0.0905 |
| garden | REJECT_RGB_REGRESSION | yes | 24.6563 | 0.7502 | 0.2521 | -0.3718 | -0.0298 | +0.0508 | -1.3785 | -0.0669 | +0.0998 |

Accepted by strict diagnostic rule: `0 / 4`.

## Geometry Delta Vs Compact-Only

| scene | dAbsRel | dDepthMAE | dNormalDeg |
|---|---|---|---|
| bicycle | -0.000757 | -0.0099 | +0.1432 |
| flowers | -0.000556 | -0.0140 | +0.1074 |
| treehill | +0.000123 | +0.0056 | +0.3336 |
| garden | -0.000054 | -0.0013 | +0.3636 |

## W&B Runs

| scene | url |
|---|---|
| bicycle | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1329lheb |
| flowers | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/a51s8z5c |
| treehill | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/qo4bse0z |
| garden | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/zmr8v9ry |

## Interpretation

This is a strict Phase-D diagnostic, not a headline method unless it
passes the table above. Negative rows are useful because they separate
representation-level recovery failures from image-space ELA gains and
prevent us from promoting a method that only looks good after test-time
post-render correction.
