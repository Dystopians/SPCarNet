# ECSR Phase-D Constrained Attribute Recovery

This report collects topology-frozen representation-level recovery runs.
The checkpoint topology is fixed, the rendered images are not edited, and
W&B is enabled during training. A run is accepted only if the recovered
checkpoint is topology-stable and does not regress PSNR, SSIM, or LPIPS
against compact-only. Compact-ELA is reported as the image-space teacher
or upper bound, not as the accepted representation-level method.

| scene | status | topology | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM vs compact | dLPIPS vs compact | dPSNR vs ELA | dSSIM vs ELA | dLPIPS vs ELA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | REJECT_RGB_REGRESSION | yes | 23.0540 | 0.6333 | 0.3853 | -0.2394 | -0.0263 | +0.0530 | -0.8587 | -0.0604 | +0.1050 |
| flowers | REJECT_RGB_REGRESSION | yes | 19.4853 | 0.4878 | 0.4278 | -0.1866 | -0.0239 | +0.0330 | -0.6975 | -0.0595 | +0.0768 |
| treehill | REJECT_RGB_REGRESSION | yes | 20.8151 | 0.5558 | 0.4483 | -0.1089 | -0.0084 | +0.0422 | -0.3833 | -0.0324 | +0.0901 |
| garden | REJECT_RGB_REGRESSION | yes | 24.6928 | 0.7503 | 0.2517 | -0.3354 | -0.0297 | +0.0504 | -1.3420 | -0.0668 | +0.0994 |

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
| bicycle | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ol7ltrp5 |
| flowers | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ko02zahe |
| treehill | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/g51aqsqo |
| garden | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/r2gkm4cq |

## Interpretation

This is a strict Phase-D diagnostic, not a headline method unless it
passes the table above. Negative rows are useful because they separate
representation-level recovery failures from image-space ELA gains and
prevent us from promoting a method that only looks good after test-time
post-render correction.
