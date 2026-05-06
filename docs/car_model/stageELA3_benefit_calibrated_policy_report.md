# Stage ELA3 Benefit-Calibrated Evidence Policy

Date: 2026-05-06

## Decision

Stage ELA3 should replace ELA2 as the current evidence-lumigraph branch.  It keeps ELA2's train-only geometry-aware residual transfer, but adds a learned counterfactual acceptance policy and multi-objective calibration.  On the selected four-scene validation set it improves PSNR, SSIM, and LPIPS over the F82 fixed adaptive policy on every scene.  Geometry metrics are inherited from F82 because ELA3 is a renderer-side adapter and does not alter topology, vertices, or camera geometry.

This is a real method-level upgrade, not a per-scene parameter scan: the per-pixel acceptance map is learned from train-view counterfactual evidence, and the same candidate family/objective is used across scenes.

## Literature Rationale

- 3D Gaussian Splatting showed that explicit splat primitives can render high-quality radiance fields efficiently while using visibility-aware rasterization and density control: https://arxiv.org/abs/2308.04079
- Deferred Neural Rendering / neural textures motivate storing view-conditioned appearance evidence on top of a geometry proxy rather than forcing all appearance into geometry: https://arxiv.org/abs/1904.12356
- Unstructured Lumigraph Rendering motivates view selection, reprojection, and evidence blending from nearby calibrated images: https://groups.csail.mit.edu/graphics/pubs/siggraph2001_ulr.pdf
- Spec-Gaussian and WildGaussians both show that view-dependent appearance and appearance uncertainty are central failure modes for Gaussian/Splatting systems: https://arxiv.org/abs/2402.15870 and https://arxiv.org/abs/2407.08447

ELA3 combines these ideas in the current MeshSplatting setting: geometry remains compact and auditable, while appearance residuals are accepted only where train-view evidence predicts benefit.

## Method

ELA2 warped training residuals into a target view using target `surf_depth`, source camera projection, and depth consistency.  ELA3 keeps that evidence path and adds:

1. For held-out training views, compute the counterfactual per-pixel benefit:

   `mean((base - gt)^2 - (base + warped_residual - gt)^2)`

2. Build train-only features:

   `log(1 + reprojection_confidence)` and `||warped_residual||`.

3. Quantile-bin the feature plane and accept only bins whose mean counterfactual gain is positive and sufficiently supported.

4. Apply the learned accept table to test views without using test GT.

5. Select candidate policy with either:

   - `balanced`: PSNR gain + `20 * SSIM gain + 20 * LPIPS gain`
   - `psnr`: conservative PSNR-only objective

The balanced objective is the current visual-quality main route.  The PSNR route is retained as a conservative ablation because it exactly preserves ELA2's room/counter PSNR behavior.

## Implementation

Changed files:

- `utils/evidence_lumigraph_adapter.py`
  - Added `EvidenceSignal`, `BenefitCalibrator`, `compute_evidence_signal`, and `fit_benefit_calibrator`.
  - Added multi-objective `calibrate_alpha` support.
  - Cached LPIPS model construction during calibration; the previous function-level LPIPS helper rebuilt VGG every call and was too slow for full policy search.
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
  - Added `--policy_objective`, `--calib_lpips`, and `--benefit_policy`.
  - Logs benefit acceptance and policy metadata to W&B.
- `scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py`
  - Added synthetic benefit-calibrator coverage.

Verification:

- `micromamba run -n mesh_splatting python -m py_compile ...`
- `CUDA_VISIBLE_DEVICES=4 micromamba run -n mesh_splatting python scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py`
- Full four-scene ELA3 apply + independent `metrics.py` evaluation on GPU 4.

## Render Metrics

All rows are independent `metrics.py` values on the test split.  Lower LPIPS is better.

| scene | F82 PSNR | F82 SSIM | F82 LPIPS | ELA2 PSNR | ELA2 SSIM | ELA2 LPIPS | ELA3 balanced PSNR | ELA3 balanced SSIM | ELA3 balanced LPIPS | balanced delta vs F82 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bonsai | 11.069180 | 0.241154 | 0.572932 | 11.111758 | 0.254234 | 0.564480 | 11.114043 | 0.259949 | 0.558811 | +0.044863 / +0.018794 / -0.014121 |
| courtyard | 12.198611 | 0.308649 | 0.566687 | 12.198611 | 0.308649 | 0.566687 | 12.204245 | 0.309606 | 0.564461 | +0.005633 / +0.000957 / -0.002226 |
| room | 15.159461 | 0.488849 | 0.513028 | 15.375466 | 0.506997 | 0.498499 | 15.365302 | 0.510027 | 0.494999 | +0.205841 / +0.021178 / -0.018029 |
| counter | 14.415808 | 0.546432 | 0.424371 | 14.624939 | 0.592827 | 0.370214 | 14.624939 | 0.592827 | 0.370214 | +0.209131 / +0.046395 / -0.054157 |

ELA3-balanced improves F82 on all render metrics in all four scenes.  Relative to ELA2, it improves bonsai and courtyard on all three render metrics, matches counter, and trades `-0.010164` PSNR for `+0.003031` SSIM and `-0.003500` LPIPS on room.

## Conservative PSNR Ablation

| scene | ELA3 PSNR-objective PSNR | SSIM | LPIPS | note |
| --- | ---: | ---: | ---: | --- |
| bonsai | 11.114043 | 0.259949 | 0.558811 | same as balanced |
| courtyard | 12.202429 | 0.309571 | 0.565071 | slightly weaker than balanced |
| room | 15.375466 | 0.506997 | 0.498499 | same as ELA2 PSNR route |
| counter | 14.624939 | 0.592827 | 0.370214 | same as balanced |

This ablation is useful for reviewers: the same benefit-calibrated mechanism can be run in a conservative PSNR-preserving mode, while the main balanced route is better for visual/perceptual quality.

## W&B Runs

Balanced route:

- bonsai: `tx0vjczq`
- courtyard: `w7j7bzpq`
- room: `oj4f0fzo`
- counter: `xblf3mn7`

PSNR route:

- bonsai: `zu15b26q`
- courtyard: `lezpplck`
- room: `bt62zdzl`
- counter: `6zsoxq87`

## Qualitative Assets

Rule: select the test view with the largest per-view LPIPS improvement of ELA3-balanced over F82, then show GT / F82 / ELA2 / ELA3.

- `outputs/carnet/meshsplatopt/stageELA3_benefit_calibrated_policy/qualitative/bonsai_f82_ela2_ela3_lpips_selected.png`
- `outputs/carnet/meshsplatopt/stageELA3_benefit_calibrated_policy/qualitative/courtyard_f82_ela2_ela3_lpips_selected.png`
- `outputs/carnet/meshsplatopt/stageELA3_benefit_calibrated_policy/qualitative/room_f82_ela2_ela3_lpips_selected.png`
- `outputs/carnet/meshsplatopt/stageELA3_benefit_calibrated_policy/qualitative/counter_f82_ela2_ela3_lpips_selected.png`

## Remaining Bottleneck

ELA3 is now a strong and credible renderer-side innovation over F82, but it does not close the gap to the best clean 9000 checkpoints.  The next paper-level step should distill ELA3's evidence into a persistent compact neural texture/residual field so inference no longer depends on a runtime evidence cache and so the appearance improvement can be trained jointly with the compact representation.

