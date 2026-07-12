# 2026-06-27 v147-v148 View-Balanced and View-Cluster MoE Linear Generator Log

## Context

The previous image-L1 bin-alpha line proved that strict image-space certification can accept a non-noop repair, but the target/test impact stayed too small to support the paper-level goal. The key failure mode was no longer only a missing gate. The method needed a representation-level change that can transfer the policy-val residual signal across views with better coverage.

This log records the next prompt-driven upgrade:

- replace purely local bin decisions with an image-space residual generator;
- train it on policy-val image residuals without using target/test GT during apply;
- balance training by view so large easy views do not dominate;
- add a view-clustered mixture-of-experts generator so different viewing directions can use different residual mappings.

## Code Changes

Files changed:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

Implemented interfaces:

- `--image_linear_generator_training_sample_policy view_balanced`
- `--image_linear_generator_training_sample_policy view_balanced_base_l1_descent`
- `--image_linear_generator_expert_mode view_cluster`
- `--image_linear_generator_expert_min_training_samples`
- `--image_linear_generator_expert_shrink_tau`

Behavior:

- `view_balanced` gives each policy-val view equal regression weight after per-view sample collection.
- `view_balanced_base_l1_descent` applies the same balancing after filtering for descent-like samples.
- `view_cluster` fits a global image-linear residual generator, then fits per-cluster experts keyed by the existing camera/view-cluster profile.
- Experts are shrunk toward the global model by `cluster_samples / (cluster_samples + shrink_tau)` so low-support clusters do not become unconstrained.
- Inference routes a target sample to its view cluster when an enabled expert exists, otherwise falling back to the global generator.
- Audit now records per-view training gain fractions, view-balanced sample weights, expert metadata, expert rows, and expert enablement.

Static verification passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

## v147 Result: View-Balanced Global Generator

Run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0830_v147_viewbalanced_linear_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0830_v147_viewbalanced_linear_flowers/wandb/offline-run-20260627_084535-yhgclisq`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- method: `ours_26000_v147_viewbalanced_linear_flowers`
- generator feature mode: `base_rgb_bary_view`
- generator loss: `huber_irls`
- generator training sample policy: `view_balanced`

Protocol:

- manifest status: completed
- manifest updated at: `2026-06-27T08:45:34`
- command count: `3`
- error count: `0`
- adapter elapsed: `2600.153` sec
- eval-GT population elapsed: `40.991` sec
- final test eval elapsed: `49.081` sec
- protocol audit: passed
- target/test GT was not visible to selection/apply

Adapter audit:

- accepted: `False`
- effective policy: `fallback_noop`
- fallback written: `True`
- selected alpha: `0.0`
- target changed pixels: `0 / 37100800`
- generator relative gain vs base MSE: `-0.0174867655`
- generator relative gain vs base L1: `0.0503798138`
- per-view training MSE gain fraction: `0.5833333333`
- per-view training L1 gain fraction: `0.6666666667`
- policy-val best alpha: `0.375`
- policy-val relative gain: `0.0843130171`
- policy-val positive-view fraction: `0.5000000000`
- policy-val SSIM positive-view fraction: `0.4166666667`
- policy-val image-L1 positive-view fraction: `0.5000000000`

Final test metrics:

| method | accepted | effective policy | changed pixels | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| v147 view-balanced global generator | no | fallback_noop | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Reject reasons:

- `positive_view_fraction 0.500000 < min_policy_val_positive_view_fraction 0.550000`
- `ssim_positive_view_fraction 0.416667 < min_policy_val_ssim_positive_view_fraction 0.550000`
- `ssim_min_view_gain -0.000017405 < min_policy_val_ssim_min_view_gain -0.000010000`
- `image_l1_positive_view_fraction 0.500000 < min_policy_val_l1_positive_view_fraction 0.550000`

Interpretation:

- This is a real improvement over the previous v146e-style failure where only about `0.1667` of views had positive policy-val benefit.
- The new view-balanced global generator lifted policy-val coverage to `0.5`, close to the strict `0.55` gate.
- It still failed the certification gate and therefore correctly fell back to no-op.
- The main remaining issue is cross-view consistency, especially SSIM coverage.

## v148 Result: View-Cluster MoE Generator

Run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0850_v148_moe_linear_flowers/flowers`
- GPU: `CUDA_VISIBLE_DEVICES=5`
- method: `ours_26000_v148_moe_linear_flowers`
- W&B mode: offline
- W&B project: `spcarnet_meshprior`
- W&B group/name: `v148_moe_linear` / `flowers_v148_moe_linear`
- W&B path: `/dev/shm/peilincai_wandb_20260627_0850_v148_moe_linear_flowers/wandb/offline-run-20260627_091148-x8h7bj11`
- generator policy: `view_balanced`
- expert mode: `view_cluster`
- expert min training samples: `1024`
- expert shrink tau: `4096`

Protocol:

- manifest status: completed
- manifest updated at: `2026-06-27T09:11:48`
- command count: `3`
- error count: `0`
- adapter elapsed: `2932.543` sec
- eval-GT population elapsed: `30.551` sec
- final test eval elapsed: `44.426` sec
- protocol audit: passed
- target/test GT was not visible to selection/apply

Adapter audit:

- accepted: `False`
- effective policy: `fallback_noop`
- fallback written: `True`
- selected alpha: `0.0`
- target changed pixels: `0 / 37100800`
- generator relative gain vs base MSE: `0.1269568060`
- generator relative gain vs base L1: `0.1232076110`
- per-view training MSE gain fraction: `0.8333333333`
- per-view training L1 gain fraction: `0.8333333333`
- expert count: `3`
- enabled expert count: `3`
- expert sample counts: `5071`, `3684`, `1721`
- expert shrink-to-expert weights: `0.5531798844`, `0.4735218509`, `0.2958569709`
- policy-val best alpha: `0.375`
- policy-val relative gain: `0.1225381840`
- policy-val positive-view fraction: `0.5000000000`
- policy-val SSIM gain: `0.0000047187`
- policy-val SSIM positive-view fraction: `0.4166666667`
- policy-val SSIM min-view gain: `-0.0000199080`
- policy-val image-L1 gain: `0.0000020402`
- policy-val image-L1 positive-view fraction: `0.5000000000`

Face-gain guard:

- candidate face count: `318`
- allowed face count: `5`
- rejected face count: `313`
- allowed sample fraction: `0.1774532264`
- decision: `reject_candidate_after_face_gain_guard`

Reject reasons:

- `positive_view_fraction 0.500000 < min_policy_val_positive_view_fraction 0.550000`
- `ssim_positive_view_fraction 0.416667 < min_policy_val_ssim_positive_view_fraction 0.550000`
- `ssim_min_view_gain -0.000019908 < min_policy_val_ssim_min_view_gain -0.000010000`
- `image_l1_positive_view_fraction 0.500000 < min_policy_val_l1_positive_view_fraction 0.550000`

Final test metrics:

| method | accepted | effective policy | changed pixels | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| v148 view-cluster MoE generator | no | fallback_noop | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- v148 is clearly stronger than v147 inside the generator fit: MSE and L1 both improve by about `12%`, and `10/12` policy-val views improve on the training residual fit.
- The final certification still fails because image-level benefits remain concentrated in only half of the validation views, while SSIM is positive on only `5/12` views.
- The face-gain guard reveals the deeper bottleneck: only `5 / 318` candidate faces are consistently safe, so most learned residual support is not reliable across views.
- Because the strict gate correctly falls back to no-op, final test metrics match the no-op reference.

## Current Assessment

There is significant method progress relative to the new prompt: the train/eval pipeline now contains a stronger cross-view residual generator with view-balanced training and view-cluster MoE routing. This is not a parameter-only change.

The expected effect has not been reached yet. v147 and v148 are both near-miss rejected runs. v148 proves the MoE generator can fit residuals substantially better, but it does not yet make a certified target/test improvement under the strict multi-view gate.

Confidence:

- Directional confidence: medium. v147's positive-view fraction jump from about `0.1667` to `0.5`, plus v148's `12%` generator-fit gain, confirm that the representation upgrade is real.
- Completion confidence: not high yet. The current method still needs a mechanism that converts good residual fitting into view-consistent, face-consistent target/test improvement.

## Next Required Checks

Next:

1. Stop treating view-balanced linear/MoE fitting alone as sufficient; it improves the residual objective but not enough views.
2. Add a support-aware expert objective or policy that penalizes faces/clusters that are not consistently positive across views during fitting, not only after fitting.
3. Consider training a small spatially shared residual field with explicit per-face/view reliability weighting, because the current post-hoc face guard discards `313 / 318` faces.
4. Add a fast mode or cached multi-view policy-val pass so future experiments do not spend about `49` minutes on one flowers adapter run.
5. Re-run flowers, then one additional scene, only after the new objective changes the per-view positive fraction above `0.55` before face guard.
