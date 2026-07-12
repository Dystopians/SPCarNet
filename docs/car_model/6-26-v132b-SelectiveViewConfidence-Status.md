# v132b Selective View-Confidence Status

Date: 2026-06-26

## Question

This log answers whether the new prompt direction has made significant progress,
whether it has reached the expected effect, and whether the current evidence is
strong enough to trust the direction.

## Implemented change

v132b adds a selective view-confidence policy to the residual texture adapter.
The policy builds a policy-val view profile from accepted and rejected validation
views, then gates residual application by positive/negative feature-kernel
confidence instead of applying one global residual atlas everywhere.

Two engineering fixes were also added:

- The runner can now reuse a pre-stripped target evidence directory through
  `--prestripped_target_evidence_dir`, avoiding repeated quota-heavy stripping
  while preserving strict no-target-GT apply.
- The view-confidence seed can fall back to the best nonzero policy-val alpha
  when the risk gate would otherwise select alpha 0, so weak scenes are tested
  instead of silently disabling the profile.

Main files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  scripts/car_model/summarize_vnext_accounting.py
git diff --check scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

Both checks passed.

## Experiments

Both runs used offline W&B logging and strict no-target-GT apply.

### flowers

Run root:

```text
/dev/shm/peilincai_spcarnet_v132b_selective_viewconf_flowers_20260626_225819
```

W&B:

```text
/dev/shm/peilincai_wandb_v132b_selective_viewconf_flowers_20260626_225819/wandb/offline-run-20260626_233607-1p6w4qph
```

Audit:

```text
/dev/shm/peilincai_spcarnet_v132b_selective_viewconf_flowers_20260626_225819/flowers/model/surface_residual_region_texture_adapter_audit.json
```

Result:

```text
accepted: false
effective_policy: fallback_noop
selected_alpha: 0.0
reject_reason: effective_relative_gain 0.000648100 < min_policy_val_effective_relative_gain 0.001000000
changed_fraction: 0.0
```

Test metrics:

```text
PSNR 20.452776
SSIM 0.549059
LPIPS 0.355544
```

Important interpretation: flowers must be counted as a v132b failure. The final
test metrics are inherited from the no-op fallback output, not from a successful
new residual application.

### counter

Run root:

```text
/dev/shm/peilincai_spcarnet_v132b_selective_viewconf_counter_20260626_225819
```

W&B:

```text
/dev/shm/peilincai_wandb_v132b_selective_viewconf_counter_20260626_225819/wandb/offline-run-20260626_234348-mbm8oe07
```

Audit:

```text
/dev/shm/peilincai_spcarnet_v132b_selective_viewconf_counter_20260626_225819/counter/model/surface_residual_region_texture_adapter_audit.json
```

Result:

```text
accepted: true
effective_policy: accepted_atlas
selected_alpha: 0.5
changed_fraction: 0.000520260
png_quantized_changed_fraction: 0.000268117
```

Best policy-val row for the accepted atlas:

```text
relative_gain: 0.0087820759
ssim_gain: 0.0000633647
image_l1_gain: 0.0000071259
```

Test metrics:

```text
PSNR 27.499756
SSIM 0.867520
LPIPS 0.238841
```

Summary table against the existing reference compare JSON:

```text
counter: dPSNR clean +0.747982, dSSIM clean +0.005465, dLPIPS clean -0.013162
counter: dPSNR v106 +0.000111, dSSIM v106 -0.000001, dLPIPS v106 -0.000006
```

Important interpretation: counter shows real local effectiveness, but the target
changed fraction is only about 0.052%, so visual impact is likely weak.

## Verdict

There is significant engineering progress, but the new prompt has not reached
the expected effect yet.

Evidence:

- Positive: the no-target-GT pipeline runs end-to-end with W&B logging and the
  new confidence mechanism can accept a real atlas on counter.
- Negative: flowers, the known weak scene, still collapses to no-op because the
  effective policy-val margin is too small.
- Negative: even the accepted counter run changes only a very small fraction of
  pixels, so this is not yet a strong visual-quality breakthrough.

Confidence in the current version is moderate for engineering correctness and
low-to-moderate for the paper-level performance claim. It is not enough to claim
that the new prompt has solved the bottleneck.

## Next required work

The next fix should not be another threshold-only scan. The likely missing piece
is representation capacity on weak scenes:

1. Add an adaptive low-support residual predictor/materializer for flowers-like
   scenes, so the method has something meaningful to apply beyond a sparse atlas.
2. Keep the policy-val no-regression certificate, but separate "no-op safety" from
   "method success" in all summaries.
3. Re-run at least flowers, counter, and one additional outdoor/texture-hard
   scene with fixed policy.
4. Only treat the direction as a milestone if weak-scene acceptance, metric gain,
   and visible changed regions all improve together.

Final status for this v132b check: NOT COMPLETE.
