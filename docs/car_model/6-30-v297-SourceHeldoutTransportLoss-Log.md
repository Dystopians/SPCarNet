# 2026-06-30 v297 Source-Heldout Transport Loss Log

This is the first implementation step after the Phase-J stall investigation
that changes the training objective rather than only adding carrier features or
policy gates.

## Motivation

The v294/v296 evidence showed that diagnostic heldout features alone do not
teach the decoder to transport Phase-J residuals across views. v297 therefore
adds an explicit train-fit-only residual transport objective:

1. Split train-fit views into source views and heldout-source views.
2. Build a source-only surface texture from the source views.
3. Sample heldout-source views during training.
4. Predict heldout residuals using the source-only texture.
5. Add this heldout prediction loss to the normal train-fit reconstruction
   objective.

This does not read target/test GT. It is a real method change in the train/eval
pipeline, but the initial pilot is not a quality breakthrough.

## Implemented Interface

File:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
```

New CLI:

```text
--enable_source_heldout_transport_loss
--source_heldout_stride
--source_heldout_loss_weight
--source_heldout_batch_fraction
--source_heldout_loss_every
--policy_val_min_changed_fraction
```

New behavior:

- `_source_heldout_transport_split` splits train-fit views into source and
  heldout subsets.
- A source-only surface texture is built and saved as
  `source_heldout_surface_feature_texture_<mode>.npz`.
- `_decoder_sample_losses` factors the pointwise residual/luma/direction/energy
  losses so the main batch and heldout batch use the same objective components.
- The training loop adds `source_heldout_loss_weight * source_heldout_loss`.
- Audit JSON, Markdown, W&B offline summaries, and stdout report source-heldout
  state.
- `--policy_val_min_changed_fraction` defaults to `1e-5`, preventing no-op
  floating-point noise from being considered policy-val all-axis success.

## Validation

Static checks:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/train_perceptual_surface_residual_decoder.py
git diff --check -- scripts/car_model/train_perceptual_surface_residual_decoder.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py --help | rg "source_heldout|policy_val_min_changed"
```

Smoke command output:

```text
/tmp/peilincai_spcarnet_v297_transport_smoke_20260630
```

Smoke facts:

| field | value |
|---|---:|
| source-heldout enabled | true |
| source views | 28 |
| heldout views | 14 |
| source-heldout loss weight | 0.35 |
| final source-heldout loss | 0.009146 |
| final source-heldout RGB loss | 0.007065 |
| final source-heldout batch cosine | -0.032846 |
| policy-val all-axis | false |
| selected alpha | 0.0 |

This confirms the new path executes and is audited.

## Pilot Comparison

Both pilot runs used flowers train-fit evidence, target exact disabled, W&B
offline, `lowrank_view_holdout_v3`, `patch_view_moe`, 32 candidate faces,
24 train steps, batch size 768, and alpha grid `0,0.03125,0.0625,0.125`.

| run | output | W&B offline |
|---|---|---|
| baseline without transport | `/tmp/peilincai_spcarnet_v297_pilot_base_20260630` | `/tmp/peilincai_spcarnet_v297_pilot_base_20260630/wandb/offline-run-20260630_164010-j8rtbt57` |
| source-heldout transport | `/tmp/peilincai_spcarnet_v297_pilot_transport_20260630` | `/tmp/peilincai_spcarnet_v297_pilot_transport_20260630/wandb/offline-run-20260630_164010-nuybeso9` |
| transport re-evaluated with fixed changed-fraction gate | `/tmp/peilincai_spcarnet_v297_pilot_transport_gatefix_20260630` | n/a |

Initial old-gate transport run showed `policy_val_all_axis_pass=true`, but this
was a false positive because the selected row had `mean_changed_fraction=0.0`.
After adding `--policy_val_min_changed_fraction`, the same checkpoint no longer
passes policy-val.

Fixed-gate pilot comparison:

| run | selected alpha | PSNR gain | SSIM gain | changed fraction | all-axis | tail-safe |
|---|---:|---:|---:|---:|---|---|
| baseline no transport | 0.12500 | -0.000000388 | +0.0000000715 | 0.000015655 | false | false |
| transport, fixed gate | 0.12500 | -0.0000000512 | +0.0000000834 | 0.000020517 | false | false |

Transport slightly reduces the PSNR negativity relative to the baseline pilot,
but the effect is still numerical-noise scale. It is not enough to run flowers
exact or full9.

## Alpha-Energy Diagnostic

To separate "too little residual energy" from "wrong residual direction", the
transport pilot checkpoint was re-evaluated without training using a wider alpha
grid:

```text
/tmp/peilincai_spcarnet_v297_transport_alpha_diag_20260630
```

| alpha | PSNR gain | SSIM gain | changed fraction |
|---:|---:|---:|---:|
| 0.000 | +0.000000000 | +0.000000000 | 0.000000000 |
| 0.125 | -0.000000051 | +0.000000083 | 0.000020517 |
| 0.250 | -0.000000329 | +0.000000167 | 0.000050166 |
| 0.500 | -0.000001632 | +0.000000310 | 0.000072225 |
| 1.000 | -0.000006476 | +0.000000513 | 0.000088947 |

The larger alphas increase changed fraction, but PSNR becomes more negative.
This means the pilot is not simply under-applied; the learned residual direction
is still not reliable enough.

## Engineering Cost

In the 24-step pilot:

- baseline train stage: about 1 minute 47 seconds;
- transport train stage: about 3 minutes 10 seconds;
- transport also needs a second source-only texture build.

The method is therefore materially more expensive. It needs a stronger signal or
a faster cached source-heldout sampler before larger runs are justified.

## Verdict

```text
Final status: NOT COMPLETE.
```

v297 is the right kind of research move because it adds a real heldout transport
objective. The initial configuration does not yet solve the Phase-J bottleneck.

Next recommended work:

1. Increase useful residual energy without relying on unsafe alpha expansion.
2. Make source-heldout batches cheaper, likely by caching sampled rows instead
   of loading NPZ views every step.
3. Add a minimum changed-fraction and positive-view requirement to all future
   promotion commands.
4. Only run target exact when policy-val has nontrivial changed fraction and
   robust positive PSNR/SSIM/LPIPS signals.
