# Hierarchical (scene-cluster) bootstrap — headline re-analysis

Primary claim form remains the pre-registered stratified mean-of-scene-means CI; the 2-stage scene-cluster interval is reported alongside (both 10k resamples, seed 0, same banked per-view arrays).

| headline | metric | stratified CI | hierarchical CI | hier. excl. 0 |
|---|---|---|---|---|
| H1_final_vs_pj2026 | psnr | +0.3607 [+0.3158,+0.4067] | +0.3607 [+0.2175,+0.5273] | YES |
| H1_final_vs_pj2026 | lpips | -0.0169 [-0.0188,-0.0149] | -0.0169 [-0.0258,-0.0098] | YES |
| H2_final_vs_primary | psnr | +1.6662 [+1.5669,+1.7658] | +1.6662 [+0.9663,+2.4481] | YES |
| H3_l6_vs_primary_anchor | psnr | +1.4877 [+1.3793,+1.5929] | +1.4877 [+0.8297,+2.2125] | YES |
| H3_l6_vs_primary_anchor | lpips | -0.0748 [-0.0775,-0.0720] | -0.0748 [-0.0882,-0.0619] | YES |
| H4_l6_vs_pj2026_b50 | psnr | +0.3362 [+0.2935,+0.3799] | +0.3362 [+0.2059,+0.4857] | YES |
| H4_l6_vs_pj2026_b50 | lpips | -0.0176 [-0.0195,-0.0157] | -0.0176 [-0.0257,-0.0112] | YES |
| H5_ate0_pj2026_vs_primary | psnr | +1.3055 [+1.2260,+1.3844] | +1.3055 [+0.7155,+1.9605] | YES |

**Verdict:** ALL headline intervals also exclude 0 under scene-cluster resampling — conclusions are robust to treating the scene as the sampling unit.
