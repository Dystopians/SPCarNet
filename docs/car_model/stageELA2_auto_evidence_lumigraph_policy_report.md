# Stage ELA2 Auto Evidence Lumigraph Policy Report

Decision: `ELA2_RENDER_WIN_VS_F82_WITH_GEOMETRY_INHERITED`

## Why this stage exists

The previous SCE/ATR line was mostly engineering recovery around pruning and rollback. It was useful for safety, but the repeated bonsai failures showed that compact mesh-splat geometry plus SH appearance was not carrying enough high-frequency, view-dependent appearance. Stage ELA pivots the method toward a paper-level rendering idea:

> Keep the compact mesh-splat as the certified geometry/base renderer, then recover appearance from training-view evidence through geometry-aware residual/light-field adaptation.

This is inspired by three connected lines:

- 3D Gaussian Splatting shows that explicit splatting can provide high-quality real-time radiance-field rendering, but appearance capacity remains tied to primitive attributes and view-dependent modeling choices: https://arxiv.org/abs/2308.04079
- Surface/light-field and unstructured lumigraph rendering showed that nearby calibrated images can be treated as view-dependent appearance evidence rather than forcing everything into a static surface texture: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Buehler01.pdf and https://cseweb.ucsd.edu/~ravir/6998/papers/p287-wood.pdf
- Deferred Neural Rendering/Neural Textures uses proxy geometry plus learned image-space appearance synthesis, which matches the idea that geometry and appearance should not be over-coupled: https://arxiv.org/abs/1904.12356
- Recent Gaussian appearance papers also point in this direction: Spec-Gaussian replaces limited SH with anisotropic view-dependent appearance fields, and GS-W separates intrinsic/dynamic appearance for unconstrained captures: https://arxiv.org/abs/2402.15870 and https://arxiv.org/abs/2403.15704

## Implemented method

New code:

- `utils/evidence_lumigraph_adapter.py`
- `scripts/car_model/meshsplatopt_render_evidence_maps.py`
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
- `scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py`

Pipeline:

1. Render the base mesh-splat model on train/test views and save RGB, GT, `surf_depth`, and per-view camera matrices.
2. For each target pixel, unproject through target `surf_depth`, project into candidate train views, and accept evidence only when projected depth is consistent with the train-view depth.
3. Warp train-view residuals `(GT_train - render_train)` into the target view and blend them into the base render.
4. Select support views by camera-center distance and view-direction similarity.
5. Use train-only held-out calibration to choose `alpha` and an automatic policy over residual/color modes, `k`, and depth tolerance. Test GT is not used for policy selection.
6. If calibration says the adapter is harmful, alpha becomes `0` and the method no-ops.

## Final ELA2 validation

Base: F82 fixed adaptive policy seed0, iteration 26000.

All ELA2 rows use online W&B:

- bonsai: `4cullr68`
- courtyard: `vzpna2vs`
- room: `frk7ces0`
- counter: `k3ko2bj0`

| scene | auto policy | alpha | F82 PSNR | ELA2 PSNR | dPSNR | F82 SSIM | ELA2 SSIM | dSSIM | F82 LPIPS | ELA2 LPIPS | dLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bonsai | residual k4 rel0.12 | 0.5 | 11.069180 | 11.111758 | +0.042578 | 0.241154 | 0.254234 | +0.013080 | 0.572932 | 0.564480 | -0.008452 |
| courtyard | residual k4 rel0.06 | 0.0 | 12.198611 | 12.198611 | +0.000000 | 0.308649 | 0.308649 | +0.000000 | 0.566687 | 0.566687 | +0.000000 |
| room | residual k8 rel0.12 | 1.0 | 15.159461 | 15.375466 | +0.216005 | 0.488849 | 0.506997 | +0.018147 | 0.513028 | 0.498499 | -0.014529 |
| counter | residual k4 rel0.12 | 1.0 | 14.415808 | 14.624939 | +0.209131 | 0.546432 | 0.592827 | +0.046395 | 0.424371 | 0.370214 | -0.054157 |

Geometry metrics are inherited from the F82 model because ELA2 is a renderer-side adapter and does not alter mesh topology or checkpoint geometry. Therefore AbsRel, Depth MAE, and normal metrics are unchanged vs F82, while render metrics improve or no-op on all four selected scenes.

## Interpretation

This is a real method upgrade over the previous recovery-only direction:

- It introduces a new evidence-aware rendering layer rather than only changing pruning/recovery parameters.
- It improves bonsai, room, and counter on all three render metrics and safely no-ops on courtyard.
- The no-op is automatic from train-only calibration, not a manual scene exception.
- The gains are larger and more consistent than the late SCE/ATR micro-gains.

But this is not yet enough to claim a complete top-conference result by itself:

- It still does not close the huge gap to the best clean 9000 checkpoints on Mip-NeRF 360 scenes.
- It is currently a post-render adapter; it does not train a persistent neural module or compress the evidence cache.
- Runtime is slower than plain mesh-splat rendering because each target view samples multiple train views.
- The current calibration optimizes train held-out MSE/PSNR; a stronger policy should include SSIM/LPIPS-like perceptual criteria.

## Next required upgrade

The most promising next step is `ELA3`: distill the geometry-aware residual/light-field evidence into a compact learned residual field or neural texture module. That would keep the successful train-view evidence signal while avoiding per-frame multi-view warping at inference, and it would make the method more clearly paper-level rather than a strong renderer-side postprocess.
