# EXP-T2B — Tanks&Temples + Deep Blending (the 3DGS eval suite)

Frozen pipeline transfer: clean30k anchor -> PJ-2026 floor -> final routed stack; per-scene paired CIs (10k, seed 0); suite means stratified AND scene-cluster.

| scene | anchor PSNR | PJ-2026 PSNR | final PSNR/LPIPS | dPSNR PJ-vs-anchor [CI] | dPSNR final-vs-PJ [CI] | dLPIPS final-vs-PJ [CI] |
|---|---|---|---|---|---|---|
| tandt_truck | 22.429 | 23.572 | 23.733/0.1512 | +1.143 [+0.995,+1.296] | +0.161 [+0.105,+0.222] | -0.011 [-0.013,-0.009] |
| tandt_train | 18.819 | 20.240 | 20.369/0.2463 | +1.421 [+0.952,+1.871] | +0.130 [-0.044,+0.322] | -0.007 [-0.013,-0.002] |
| db_drjohnson | 26.964 | 27.200 | 27.298/0.3132 | +0.236 [+0.181,+0.298] | +0.097 [+0.027,+0.177] | -0.014 [-0.019,-0.010] |
| db_playroom | 27.928 | 28.050 | 28.148/0.2744 | +0.122 [+0.078,+0.170] | +0.098 [+0.050,+0.151] | -0.001 [-0.004,+0.002] |

**Suite-4 means:**

- PJ-2026 vs anchor dPSNR: stratified **+0.7306 [+0.6050,+0.8502]**; scene-cluster +0.7306 [+0.1695,+1.3118]
- final vs PJ-2026 dPSNR: stratified **+0.1215 [+0.0700,+0.1756]**; scene-cluster +0.1215 [+0.0634,+0.1846]
- final vs PJ-2026 dLPIPS: stratified **-0.0084 [-0.0105,-0.0064]**; scene-cluster -0.0084 [-0.0134,-0.0032]
- final vs anchor dPSNR: stratified **+0.8520 [+0.7491,+0.9544]**; scene-cluster +0.8520 [+0.2645,+1.4497]

Coverage: tandt_truck 0.943, tandt_train 0.943, db_drjohnson 0.668, db_playroom 0.492
