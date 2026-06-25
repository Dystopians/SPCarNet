# v100 Checkpoint-Attached ELA Full9 Summary

- output root: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625`
- all return codes zero: `True`
- all scene gates pass: `True`
- overall pass: `True`
- claim boundary: `Phase-J sidecar replay/materialization, not an independent improvement over Phase-J`

| scene | status | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR legacy source | dSSIM legacy source | dLPIPS legacy source | dPSNR Phase-J | dSSIM Phase-J | dLPIPS Phase-J |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | PASS_COUNTER_GATE | 24.021544 | 0.702357 | 0.266088 | +0.719931 | +0.042489 | -0.065989 | +0.108862 | +0.008642 | -0.014180 | +0.000000 | +0.000000 | +0.000000 |
| flowers | PASS_COUNTER_GATE | 20.300608 | 0.557458 | 0.329505 | +0.618351 | +0.045636 | -0.065058 | +0.117828 | +0.010157 | -0.021487 | -0.003750 | -0.000312 | +0.000283 |
| garden | PASS_COUNTER_GATE | 26.311111 | 0.827843 | 0.135843 | +1.281900 | +0.047808 | -0.065472 | +0.276281 | +0.010731 | -0.016469 | +0.000000 | +0.000000 | +0.000000 |
| stump | PASS_COUNTER_GATE | 25.595104 | 0.724074 | 0.263909 | +0.390062 | +0.018909 | -0.030095 | +0.232574 | +0.011545 | -0.017840 | +0.000000 | +0.000000 | +0.000000 |
| treehill | PASS_COUNTER_GATE | 21.296227 | 0.595606 | 0.336319 | +0.362045 | +0.031083 | -0.069725 | +0.097832 | +0.007414 | -0.021805 | +0.000000 | +0.000000 | +0.000000 |
| room | PASS_COUNTER_GATE | 30.305639 | 0.905730 | 0.195989 | +1.558363 | +0.020887 | -0.053913 | +1.174671 | +0.020848 | -0.052740 | +0.000000 | +0.000000 | +0.000000 |
| counter | PASS_COUNTER_GATE | 28.449171 | 0.893731 | 0.186472 | +1.697397 | +0.031675 | -0.065531 | +1.208748 | +0.029586 | -0.063229 | +0.000000 | +0.000000 | +0.000000 |
| kitchen | PASS_COUNTER_GATE | 30.199732 | 0.916087 | 0.131955 | +2.381180 | +0.039635 | -0.067231 | +2.200171 | +0.039169 | -0.066995 | +0.000000 | +0.000000 | +0.000000 |
| bonsai | PASS_COUNTER_GATE | 31.862005 | 0.930280 | 0.172555 | +2.966772 | +0.033879 | -0.086937 | +2.077568 | +0.032111 | -0.084846 | +0.000000 | +0.000000 | +0.000000 |
