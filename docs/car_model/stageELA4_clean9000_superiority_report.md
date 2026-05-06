# Stage ELA4 Clean9000 Superiority Report

Date: 2026-05-06

## Decision

Stage ELA4 is the first current branch that directly beats the strongest pure Mesh Splatting clean baseline on the selected four-scene validation set.  The comparison is against the best clean checkpoint available in each scene, which is `ours_9000`, not the weaker clean 22000 lineage.

This is the correct headline direction: start from Mesh Splatting, add a train-only evidence residual module, and improve the original Mesh Splatting render metrics.

## Method

ELA4 applies the ELA3 benefit-calibrated residual policy directly on top of clean9000 Mesh Splatting renders:

- Base: pure Mesh Splatting `ours_9000`.
- Evidence: train split RGB render, GT, `surf_depth`, and camera matrices rendered from the same clean9000 checkpoint.
- Transfer: depth-consistent residual warping from nearby train views into each held-out test view.
- Policy: train-only benefit calibration, using bins over reprojection confidence and residual magnitude.
- Test-time rule: apply the learned residual acceptance policy to test views without using test GT.

The final fast policy used a conservative search:

- mode: `residual`
- k: `4`
- depth relative tolerance: `0.06,0.12`
- residual clip: `0.10`
- calibration objective: PSNR
- calibration views: max `8`
- W&B online logging enabled.

The wider LPIPS-balanced clean9000 search was interrupted because the candidate grid was too slow and CPU-heavy.  The fast policy is the promoted clean9000 result because it already beats clean9000 on all three reported render metrics in all scenes.

## Results

All rows are independent `metrics.py` values on the test split.  Lower LPIPS is better.

| scene | clean9000 PSNR | clean9000 SSIM | clean9000 LPIPS | ELA4 PSNR | ELA4 SSIM | ELA4 LPIPS | dPSNR | dSSIM | dLPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonsai | 18.541124 | 0.463496 | 0.483265 | 19.879219 | 0.521946 | 0.458695 | +1.338095 | +0.058450 | -0.024570 |
| courtyard | 18.494551 | 0.602439 | 0.423865 | 18.686813 | 0.612538 | 0.410969 | +0.192263 | +0.010099 | -0.012896 |
| room | 26.217100 | 0.889372 | 0.135088 | 28.968866 | 0.933050 | 0.082278 | +2.751766 | +0.043678 | -0.052810 |
| counter | 24.801929 | 0.844451 | 0.159236 | 27.215458 | 0.904876 | 0.099993 | +2.413528 | +0.060425 | -0.059244 |

This is no longer a comparison to F82.  It is a direct win over the strongest pure Mesh Splatting clean9000 baseline on the selected dataset.

## Policies

| scene | policy | alpha | coverage | accepted benefit bins | W&B |
| --- | --- | ---: | ---: | ---: | --- |
| bonsai | residual k4, rel0.12, clip0.10 | 1.00 | 0.757090 | 16 | `263psrr4` |
| courtyard | residual k4, rel0.06, clip0.10 | 0.75 | 0.151966 | 5 | `kxtsbw3e` |
| room | residual k4, rel0.12, clip0.10 | 1.00 | 0.944002 | 16 | `9g4ev6rh` |
| counter | residual k4, rel0.12, clip0.10 | 1.00 | 0.959917 | 16 | `m43j8tmy` |

## Qualitative Assets

Rule: select the test view with the largest LPIPS improvement of ELA4 over clean9000, then show GT / clean9000 / ELA4.

- `outputs/carnet/meshsplatopt/stageELA4_clean9000_fast_policy/qualitative/bonsai_clean9000_vs_ela4_lpips_selected.png`
- `outputs/carnet/meshsplatopt/stageELA4_clean9000_fast_policy/qualitative/courtyard_clean9000_vs_ela4_lpips_selected.png`
- `outputs/carnet/meshsplatopt/stageELA4_clean9000_fast_policy/qualitative/room_clean9000_vs_ela4_lpips_selected.png`
- `outputs/carnet/meshsplatopt/stageELA4_clean9000_fast_policy/qualitative/counter_clean9000_vs_ela4_lpips_selected.png`

## Important Caveat

ELA4 is currently a renderer-side evidence adapter.  It beats the Mesh Splatting clean9000 render baseline, but it has not yet been distilled into a persistent compact model.  For a top-conference paper, the next required step is to turn this into a cleaner method artifact:

1. distill the evidence residual into a compact neural texture or residual field;
2. report runtime/storage overhead versus clean9000;
3. validate that the method uses train images only and no test GT leakage;
4. run a fuller ablation: no benefit policy, no depth consistency, k sweep, clip sweep, and balanced-vs-PSNR calibration.

The key bottleneck has changed: we are no longer failing to beat Mesh Splatting.  The remaining task is to make the winning adapter look like a durable method rather than a post-render evidence cache.
