# 2026-06-27 v141 Image-L1 Bin Alpha Optimizer Log

## Why this change exists

v138-v140 showed that continuing to expand uncertainty-shrink gates can find a few locally positive bins, but the final policy-val gain remains microscopic and the strict gate often falls back to no-op. The core weakness is that those variants still optimize residual proxy evidence instead of the actual rendered image error.

v141 changes the local residual policy from proxy shrink to direct image-space optimization:

- fit the usual face/UV residual atlas from train evidence;
- on train policy-val views only, evaluate `clip(rgb_render + alpha * residual_delta)` against `rgb_gt`;
- choose a local scalar alpha per `(face, UV-bin)` by minimizing policy-val image L1;
- store only bins with positive image-L1 evidence and sufficient positive-view fraction;
- set uncertified bins to no-op by `fallback_mode=zero`;
- keep the existing strict policy-val risk gate and target/test-GT-free apply path.

This is still not the final trainable generator, but it is a real method change: the acceptance signal is image-space L1, not residual MSE or manual parameter scanning.

## Code changes

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
  - Added `calibrated_image_l1_bin_alpha_profile_from_policy_val`.
  - Added CLI:
    - `--enable_policy_val_image_l1_bin_alpha_optimization`
    - `--image_l1_bin_alpha_grid`
    - `--image_l1_bin_alpha_max_alpha`
    - `--image_l1_bin_alpha_min_bin_samples`
    - `--image_l1_bin_alpha_min_relative_gain`
    - `--image_l1_bin_alpha_min_positive_view_fraction`
    - `--image_l1_bin_alpha_count_tau`
    - `--image_l1_bin_alpha_fallback_mode`
    - `--image_l1_bin_alpha_max_profile_bins`
  - Reuses existing `policy_val_bin_alpha` profile mode, so `predict_delta_for_npz`, `evaluate_policy_val`, and `apply_to_target` stay on the same inference contract.
  - Audit explicitly records `uses_policy_val_gt=True` and `uses_target_or_test_gt=False`.
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
  - Added runner CLI and adapter forwarding.
  - When the new image-L1 optimizer is enabled, the runner no longer forwards `--enable_policy_val_bin_uncertainty_shrink`, so the new method is a replacement local-alpha policy rather than a stacked gate.

## Verification

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

Both checks passed.

Dry-run:

- root: `/dev/shm/peilincai_spcarnet_20260627_043227_v141_dryrun/flowers`
- W&B: `wandb/offline-run-20260627_043308-zqmqlkpp`
- protocol audit passed;
- adapter command included `--enable_policy_val_image_l1_bin_alpha_optimization`;
- adapter command did not include `--enable_policy_val_bin_uncertainty_shrink`.

## v140 negative baseline for this change

v140a region expansion:

- root: `/dev/shm/peilincai_spcarnet_v140a_region_expand_flowers_20260627_0404/flowers`
- accepted: `False`
- effective policy: `fallback_noop`
- selected bins: `60`
- policy-val relative gain: `0.000017421`
- test metrics: PSNR `20.452775955`, SSIM `0.549059212`, LPIPS `0.355544209`

v140b region expansion + sparse materialization:

- root: `/dev/shm/peilincai_spcarnet_v140b_region_expand_sparse_flowers_20260627_0404/flowers`
- accepted: `False`
- effective policy: `fallback_noop`
- selected bins: `60`
- policy-val relative gain: `0.000033323`
- test metrics: PSNR `20.452775955`, SSIM `0.549059212`, LPIPS `0.355544209`

Conclusion: v140 increased local selected bins but never crossed the effective policy-val margin, so it made no target render changes.

## v141 result

Command group:

- root: `/dev/shm/peilincai_spcarnet_20260627_0435_v141_image_l1_bin_alpha_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0435_v141_image_l1_bin_alpha_flowers/wandb/offline-run-20260627_045121-u5yz3o7a`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- strict target/test GT apply: enabled

Key audit:

- accepted: `True`
- effective policy: `accepted_atlas`
- selected alpha: `0.125`
- local alpha mode: `policy_val_bin_alpha`
- optimizer: `policy_val_image_l1_grid`
- policy-val GT used for selection: `True`
- target/test GT used for selection or apply: `False`
- candidate optimized bins: `213`
- stored bin alphas: `213`
- fallback alpha: `0.0`
- mean selected bin image-L1 relative gain: `0.128531263`
- policy-val relative gain: `0.008921483`
- policy-val positive-view fraction: `0.75`
- target changed pixels: `1885 / 37100800`
- PNG-quantized changed pixels: `252 / 37100800`

Test metrics:

| method | accepted | changed pixels | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v140a no-op | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |
| v140b no-op | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |
| v141 image-L1 bin alpha | yes | 1885 | 20.452779770 | 0.549059153 | 0.355544418 |

## Interpretation

v141 is an engineering and methodological milestone because it is the first strict run in this branch that:

- changes the method rather than only adding rejection gates;
- directly optimizes image-space L1 on policy-val views;
- passes the strict gate instead of falling back to no-op;
- writes nonzero target render changes without using target/test GT for selection.

It is not yet a paper-level success:

- target changed-pixel coverage is only `0.0051%`;
- PSNR gain is extremely small;
- SSIM and LPIPS do not improve on the flowers test split;
- this still supports only a weak claim: image-space optimization can pass the strict certification loop, but coverage and perceptual impact are insufficient.

## v142 coverage expansion result

v142 tested whether the same optimizer becomes materially useful when policy-val supervision coverage is expanded:

- root: `/dev/shm/peilincai_spcarnet_20260627_0453_v142_image_l1_bin_alpha_cov_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0453_v142_image_l1_bin_alpha_cov_flowers/wandb/offline-run-20260627_051238-xfr0gci2`
- GPU: `CUDA_VISIBLE_DEVICES=5`
- key changes: `adaptive_residual_activity_floor=0.003`, `adaptive_residual_activity_quantile=0.85`, `image_l1_bin_alpha_min_bin_samples=2`, positive-view fraction `0.55`, max profile bins `16384`.

Key audit:

- accepted: `True`
- effective policy: `accepted_atlas`
- selected alpha: `0.5`
- local alpha mode: `policy_val_bin_alpha`
- optimizer: `policy_val_image_l1_grid`
- policy-val GT used for selection: `True`
- target/test GT used for selection or apply: `False`
- candidate optimized bins: `741`
- stored bin alphas: `741`
- fallback alpha: `0.0`
- mean selected bin image-L1 relative gain: `0.160311269`
- policy-val relative gain: `0.029949560`
- policy-val image-L1 gain: `0.000000234`
- policy-val SSIM gain: `0.000000273`
- policy-val LPIPS gain: `0.000001591`
- target changed pixels: `1186 / 37100800`
- PNG-quantized changed pixels: `547 / 37100800`

Test metrics:

| method | accepted | changed pixels | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v140 no-op reference | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |
| v141 image-L1 bin alpha | yes | 1885 | 20.452779770 | 0.549059153 | 0.355544418 |
| v142 image-L1 bin alpha coverage | yes | 1186 | 20.452791214 | 0.549059212 | 0.355544418 |

## Updated interpretation

v142 is stronger than v141 as a strict policy-val certification result: the selected policy-val relative gain rose from `0.008921483` to `0.029949560`, and the run still used no target/test GT for selection or apply.

It still does not meet the expected paper-level effect:

- target coverage stayed extremely small: `0.0032%` changed pixels and `0.0015%` PNG-quantized changed pixels;
- PSNR improved only at numerical-noise scale;
- SSIM was essentially unchanged;
- LPIPS did not beat the no-op reference;
- qualitative improvement is unlikely to be visible because most target pixels are untouched.

The new prompt has therefore produced a useful bottleneck diagnosis rather than a finished method. The current local scalar policy can find certified image-space improvements on policy-val views, but the certified signal does not propagate with enough spatial support to create a meaningful target/test improvement.

## Immediate next step

Run one targeted diagnostic, v143, to test whether the final policy-val bin uncertainty guard is the main reason target coverage collapses after v142. v143 should keep the same image-L1 bin-alpha optimizer and strict no-target-GT protocol, but set `--no_policy_val_bin_uncertainty_guard`. If v143 increases target changed-pixel coverage without improving test metrics, the limitation is not just guard conservatism; the next method step should be a true trainable image-space residual generator instead of more per-bin scalar selection.

## v143 no-bin-guard diagnostic

v143 completed the targeted guard diagnostic:

- root: `/dev/shm/peilincai_spcarnet_20260627_0517_v143_image_l1_bin_alpha_noguard_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0517_v143_image_l1_bin_alpha_noguard_flowers/wandb/offline-run-20260627_054057-9xw9pf98`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- protocol audit: passed
- command count: `3`
- error count: `0`
- main diagnostic change vs v142: `--no_policy_val_bin_uncertainty_guard`

Key audit:

- accepted: `True`
- effective policy: `accepted_atlas`
- selected alpha: `0.5`
- local alpha optimizer: `policy_val_image_l1_grid`
- policy-val relative gain: `0.029949560`
- policy-val image-L1 gain: `0.000000234`
- policy-val SSIM gain: `0.000000273`
- policy-val LPIPS gain: `0.000001591`
- target changed pixels: `1186 / 37100800`
- PNG-quantized changed pixels: `547 / 37100800`

Test metrics:

| method | diagnostic change | accepted | changed pixels | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v142 | bin guard enabled | yes | 1186 | 20.452791214 | 0.549059212 | 0.355544418 |
| v143 | bin guard disabled | yes | 1186 | 20.452791214 | 0.549059212 | 0.355544418 |

Conclusion: the tiny target/test effect is not caused by the final bin uncertainty guard. v143 reproduced v142 exactly at the observable metric and changed-pixel level. The current failure mode is therefore upstream: the residual generator/atlas has certified local policy-val wins, but the learned residual support is too sparse and too weak on target-visible regions to move full-image metrics or qualitative appearance.

## Current confidence after v143

- Confidence that the current v141-v143 scalar image-L1 bin-alpha policy is a finished paper-level method: low.
- Confidence that the new prompt produced a meaningful diagnosis: high.
- Confidence in the next direction: medium-high. The next real method upgrade should stop adding gates around the same residual field and instead train a stronger image-space residual generator with explicit policy-val certification, cached policy-val renders, and target/test-GT-free application.

## v144/v144b image-linear residual generator

v144 implemented the next proposed method upgrade: a policy-val-trained ridge
linear image-space residual generator.  Instead of only scaling the existing
surface residual by a per-bin alpha, the generator predicts
`rgb_gt - rgb_render` from target-available features:

- bias;
- current atlas residual RGB and residual magnitude;
- parent render RGB and luminance.

The implementation is target/test-GT-free at apply time.  It stores
`uses_policy_val_gt=True` and `uses_target_or_test_gt=False` in the local alpha
profile, then lets the existing policy-val risk gate select alpha or reject to
no-op.

Code paths:

- generator feature/apply helper:
  `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- policy-val ridge fit:
  `calibrated_image_linear_generator_profile_from_policy_val`
- runner forwarding:
  `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

The first v144 run exposed a performance bug: each face-local generator apply
re-read or re-cast the full `rgb_render` image.  It was interrupted after the
first policy-val view took about `143s`.  v144b fixed this by caching
per-view render/luma/camera features inside `predict_delta_for_npz`; the same
strict run then completed successfully.

v144b run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0559_v144b_image_linear_generator_cache_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0559_v144b_image_linear_generator_cache_flowers/wandb/offline-run-20260627_062716-6i789uig`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- status: `COMPLETE`
- command count: `3`
- error count: `0`
- protocol audit passed: `True`
- target GT visible to apply: `False`
- target GT visible to selection: `False`
- selection uses test GT: `False`

Generator fit summary:

- feature mode: `base_rgb`
- ridge: `0.01`
- samples: `10476`
- policy-val views used: `12`
- generator MSE gain vs base residual: `+0.050989576`
- generator MSE gain vs zero residual: `+0.054925076`
- generator L1 gain vs base residual: `-0.006419136`
- generator L1 gain vs zero residual: `-0.016646378`

Policy-val decision:

- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- candidate relative MSE gain: `0.028183487`
- candidate SSIM gain: `-0.000013813376`
- candidate image-L1 gain: `-0.000002685003`
- target changed pixels: `0 / 37100800`

Test metrics:

| method | accepted | changed pixels | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v143 image-L1 bin alpha no-guard | yes | 1186 | 20.452791214 | 0.549059212 | 0.355544418 |
| v144b image-linear generator | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- v144b is a real train/eval method change, not a parameter scan.
- The strict protocol behaved correctly: the generator found an MSE-improving
  residual field on policy-val samples, but it made image-L1 and SSIM worse,
  so the gate rejected it before target application.
- This explains why flowers remains difficult: optimizing residual-vector MSE
  or linear image residuals is not aligned enough with perceptual/structural
  image quality on thin vegetation and high-frequency outdoor content.
- The next method step should be structure-aware at the generator objective
  level, not only at the post-hoc gate level.  A plausible next attempt is a
  policy-val-trained local generator or basis that directly optimizes image-L1
  plus gradient/SSIM proxy under small target-visible support, then performs
  the same no-target-GT certified apply.  A pure global linear ridge model is
  too weak and too MSE-biased for this failure mode.

Current confidence after v144b:

- Confidence that v144b is finished paper-level progress: low.
- Confidence that it gives useful negative evidence: high.
- Confidence that the new prompt has improved the protocol and diagnosis:
  high.
- Confidence that the current vNext residual-texture family alone will produce
  a large visual jump without a stronger structure-aware representation:
  low-to-medium.

## v145 robust image-linear generator

v145 tested whether the v144 failure was mainly caused by an MSE-biased global
ridge objective.  The generator was upgraded with:

- L1-IRLS robust fitting;
- `base_l1_descent` training-sample filtering, so the model learns only from
  policy-val samples where the original atlas residual already reduces
  image-space L1 relative to no residual;
- explicit audit fields for raw samples, kept samples, rejected samples, IRLS
  history, and train-policy-val/target-GT provenance.

Run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0632_v145_l1irls_descent_generator_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0632_v145_l1irls_descent_generator_flowers/wandb/offline-run-20260627_070041-fhe6r7ih`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- status: `COMPLETE`
- command count: `3`
- error count: `0`
- protocol audit passed: `True`
- target GT visible to apply: `False`
- target GT visible to selection: `False`
- selection uses test GT: `False`

Generator fit summary:

- feature mode: `base_rgb`
- loss mode: `l1_irls`
- training sample policy: `base_l1_descent`
- raw samples: `10476`
- kept samples: `5531`
- rejected samples: `4945`
- keep fraction: `0.527968690`
- generator MSE gain vs base residual: `+0.353019020`
- generator L1 gain vs base residual: `+0.220079372`

Policy-val decision:

- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- reject reason included negative worst-view MSE, SSIM, and image-L1 gates;
- target changed pixels: `0 / 37100800`

Test metrics:

| method | accepted | changed pixels | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v143 image-L1 bin alpha no-guard | yes | 1186 | 20.452791214 | 0.549059212 | 0.355544418 |
| v144b image-linear generator | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |
| v145 L1-IRLS descent generator | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- v145 proves that the generator can fit local residual samples much better
  than v144b, including under an L1-like objective.
- The full-image gate still rejects it.  This is the important failure signal:
  local sample residual quality is not sufficient evidence for multi-view
  image-level improvement.
- The next useful test should therefore return to image-level certification
  and increase its support/coverage, rather than improving local residual
  regression alone.

## v146b image-L1 bin alpha revisit

Purpose:

- revisit the strongest strict accepted branch, `policy_val_image_l1_grid`;
- keep the v145 strict no-target-GT protocol and W&B logging;
- disable the replacement generator;
- use `fallback_mode=zero` so uncertified bins are no-op;
- use a denser local alpha grid and lower `min_bin_samples=4` to test whether
  support coverage can increase without leaking target/test GT.

Run launched:

- root pattern: `/dev/shm/peilincai_spcarnet_20260627_*_v146b_image_l1_bin_alpha_flowers/flowers`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- W&B mode: `offline`
- method: `ours_26000_v146b_image_l1_bin_alpha_flowers`

Fields to fill after completion:

- manifest status, command count, error count, protocol audit;
- adapter `accepted`, `effective_policy`, `selected_alpha`,
  `reject_reason`;
- local alpha profile `candidate_bin_count`, `bin_alpha_count`,
  `global_relative_gain`, `mean_selected_relative_gain`, fallback alpha;
- target `changed_pixels`, `png_quantized_changed_pixels`, total pixels;
- test PSNR, SSIM, LPIPS;
- whether the run improves over v143 or only reproduces the same sparse
  effect.

## v146c/v146d strict revisit results

v146b was interrupted manually after it appeared silent because runner captures
subprocess stdout into log files.  This was an operator mistake, not a code
failure.  The same experiment was restarted as v146c and allowed to complete.

### v146c: image-L1 bin alpha with LPIPS gate

Run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0708_v146c_image_l1_bin_alpha_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0708_v146c_image_l1_bin_alpha_flowers/wandb/offline-run-20260627_072651-n4xl7jet`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- status: `COMPLETE`
- command count: `3`
- error count: `0`
- protocol audit passed: `True`

Key audit:

- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- local alpha optimizer: `policy_val_image_l1_grid`
- candidate bins: `221`
- stored bin alphas: `221`
- fallback bins: `6600`
- mean selected bin image-L1 relative gain: `0.130323980`
- policy-val best alpha before rejection: `1.0`
- policy-val relative gain: `0.079996480`
- policy-val positive-view fraction: `0.75`
- policy-val SSIM gain: `0.000001952`
- policy-val image-L1 gain: `0.000000902`
- policy-val LPIPS gain: `-0.000004407`
- LPIPS positive-view fraction: `0.333333`
- target changed pixels after fallback: `0 / 37100800`

Reject reason:

- `lpips_gain -0.000004407 < min_policy_val_lpips_mean_gain 0.000000000`
- `lpips_positive_view_fraction 0.333333 < min_policy_val_lpips_positive_view_fraction 0.550000`

Test metrics:

| method | LPIPS gate | prior hybrid | accepted | changed pixels | PSNR | SSIM | LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| v146c image-L1 bin alpha | on | on | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- The image-L1 bin-alpha candidate is positive under residual MSE, SSIM, and
  image-L1 policy-val metrics.
- It is not perceptually safe under LPIPS, so the strict visual gate correctly
  rejects it.
- This result argues against claiming visual-quality improvement from this
  branch unless a later variant also passes LPIPS or visibly improves
  qualitative renders.

### v146d: image-L1 bin alpha without LPIPS gate

Run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0712_v146d_image_l1_bin_alpha_nolpips_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0712_v146d_image_l1_bin_alpha_nolpips_flowers/wandb/offline-run-20260627_073411-h4g57bsd`
- GPU: `CUDA_VISIBLE_DEVICES=5`
- status: `COMPLETE`
- command count: `3`
- error count: `0`
- protocol audit passed: `True`

Key audit:

- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- local alpha optimizer: `policy_val_image_l1_grid`
- candidate bins: `221`
- stored bin alphas: `221`
- fallback bins: `6600`
- mean selected bin image-L1 relative gain: `0.130323980`
- policy-val best alpha before rejection: `1.0`
- policy-val relative gain: `0.004736623`
- policy-val positive-view fraction: `0.166667`
- policy-val SSIM gain: `0.000000258`
- policy-val image-L1 gain: `0.000000044`
- target changed pixels after fallback: `0 / 37100800`

Reject reason:

- `positive_view_fraction 0.166667 < min_policy_val_positive_view_fraction 0.550000`
- `ssim_positive_view_fraction 0.166667 < min_policy_val_ssim_positive_view_fraction 0.550000`
- `image_l1_positive_view_fraction 0.166667 < min_policy_val_l1_positive_view_fraction 0.550000`

Test metrics:

| method | LPIPS gate | prior hybrid | accepted | changed pixels | PSNR | SSIM | LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| v146d image-L1 bin alpha no-LPIPS | off | on | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- Removing LPIPS alone does not rescue the method when the default prior-bin
  hybrid/source-mixture branch is enabled.
- The final selected/mixed candidate regresses in view coverage: only `2/12`
  policy-val views are positive.
- This suggests a new engineering-method gap: the prior-hybrid branch can
  overwrite or re-rank away from a stronger primary image-L1 bin-alpha
  candidate.  A fast ablation should disable prior-hybrid and test the primary
  candidate directly.

## v146e fast no-prior ablation

To test the hypothesis above, the runner now exposes:

```bash
--no_policy_val_prior_bin_gain_hybrid
```

This keeps the primary certified atlas candidate and target-support ranking but
skips the expensive prior-bin-gain/source-mixture hybrid branch.  v146e ran
with this flag and without LPIPS gate to isolate whether the primary image-L1
bin-alpha candidate can be accepted and written to target/test.

Run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0735_v146e_image_l1_bin_alpha_noprior_nolpips_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0735_v146e_image_l1_bin_alpha_noprior_nolpips_flowers/wandb/offline-run-20260627_075638-4tg6lqwy`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- status: `COMPLETE`
- command count: `3`
- error count: `0`
- protocol audit passed: `True`

Key audit:

- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- local alpha optimizer: `policy_val_image_l1_grid`
- candidate bins: `221`
- stored bin alphas: `221`
- fallback bins: `6600`
- mean selected bin image-L1 relative gain: `0.130323980`
- policy-val best alpha before rejection: `1.0`
- policy-val relative gain: `0.004736623`
- policy-val positive-view fraction: `0.166667`
- policy-val SSIM gain: `0.000000258`
- policy-val image-L1 gain: `0.000000044`
- target changed pixels after fallback: `0 / 37100800`

Reject reason:

- `positive_view_fraction 0.166667 < min_policy_val_positive_view_fraction 0.550000`
- `ssim_positive_view_fraction 0.166667 < min_policy_val_ssim_positive_view_fraction 0.550000`
- `image_l1_positive_view_fraction 0.166667 < min_policy_val_l1_positive_view_fraction 0.550000`

Test metrics:

| method | LPIPS gate | prior hybrid | accepted | changed pixels | PSNR | SSIM | LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| v146e image-L1 bin alpha no-prior no-LPIPS | off | off | no | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- Disabling the prior-bin-gain/source-mixture hybrid does not rescue the
  candidate.  v146d and v146e share the same global policy-val profile, so the
  main blocker is not just prior-hybrid negative transfer.
- The decisive weakness is cross-view coverage: only `2/12` policy-val views
  are positive under the residual, SSIM, and image-L1 gates.
- The current image-L1 bin-alpha branch is therefore a useful diagnostic and
  safety mechanism, but not yet a successful visual-quality improvement.
  Further work should change the residual generator/representation so positive
  gains are consistent across views, rather than relaxing gates or searching
  more scene-specific parameters.
