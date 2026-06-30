# 2026-06-30 v293 TextureBinLatent PatchViewMoE v169 Gate Log

This log follows:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

The binding gate remains:

```text
Phase-J flowers: PSNR > 20.304358, SSIM > 0.557770, LPIPS < 0.329222
```

No full9 was launched because the flowers Phase-J gate still fails.

## Motivation

v292d fixed the main v290 failure: raw PatchViewMoE improved PSNR but damaged
target SSIM/LPIPS, while the target-blind view-support gate restored target
all-axis gains versus the parent. The remaining gap was about `0.452906 dB`
below the Phase-J flowers PSNR gate.

v293 therefore changes carrier capacity instead of scanning alpha or gate
thresholds.

## Implementation

File:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
```

New mechanism:

- per face/UV-bin trainable texture latent embeddings;
- latent is concatenated to the existing geometric/view/surface-feature rows;
- the same latent index path is used in train batches, image proxy loss,
  policy-val, best render writing, and target no-GT exact apply;
- optional partial checkpoint loading expands the first MLP layer so older
  PatchViewMoE checkpoints can be warm-started while zero-initializing new
  latent input columns.

New CLI:

```text
--texture_latent_dim
--texture_latent_init_std
--texture_latent_reg
--allow_partial_init_checkpoint
```

This is a real representation-level change: the surface representation now has
a learned neural texture capacity beyond static low-rank residual statistics.

## Experiments

Both runs fixed the v292d view-support policy:

```text
--view_support_gate_mode lowrank_view_cos
--view_support_min_cos 0.95
--view_support_min_concentration 0.15
--view_support_power 1.0
--view_support_floor 0.35
--alpha_grid 0.75
```

W&B was enabled offline:

```text
outputs/carnet/spcarnet_v293_texture_latent_20260630/v293a_patch_view_moe_texture_latent_dim8_fixedviewsupport_20260630/wandb/offline-run-20260630_151008-51sked7q
outputs/carnet/spcarnet_v293_texture_latent_20260630/v293b_warmstart_texture_latent_dim4_fixedviewsupport_20260630/wandb/offline-run-20260630_151401-f8flt68s
```

## Results

Reference before v293:

| run | target PSNR | target SSIM | target LPIPS | PSNR gain | SSIM gain | LPIPS gain | Phase-J PSNR gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| v292d | 19.851452 | 0.620343 | 0.180212 | +0.019398 | +0.000432 | +0.000123 | -0.452906 |

v293 results:

| run | init | target PSNR | target SSIM | target LPIPS | PSNR gain | SSIM gain | LPIPS gain | changed | Phase-J PSNR gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v293a latent dim 8 | scratch | **19.853420** | 0.620328 | 0.180312 | **+0.021366** | +0.000418 | +0.000022 | 0.112679 | **-0.450938** |
| v293b latent dim 4 | v290 warm-start | 19.852988 | **0.620345** | 0.180246 | +0.020934 | **+0.000435** | +0.000088 | 0.113481 | -0.451370 |

No-target-GT audits passed for both v293 exact runs.

## Interpretation

v293 is an important diagnostic:

- The learned texture latent does increase carrier capacity.
- v293a gives the best flowers target PSNR seen so far in this branch.
- v293b proves partial warm-start from v290 PatchViewMoE works:
  `net.0.weight` expanded from input dim `116` to `120`, with no skipped keys.
- Both runs still fail tail safety on target. SSIM and LPIPS tails remain
  negative.
- Neither run beats v292d all-axis because LPIPS is worse than v292d.
- Neither run passes the Phase-J flowers PSNR gate.

The current best PSNR frontier is v293a. The current best balanced frontier
remains v292d. The bottleneck has shifted: adding neural texture capacity helps
PSNR, but it increases target-view perceptual tail risk.

## Artifacts

```text
docs/car_model/results/v293_texture_latent_summary.json
outputs/carnet/spcarnet_v293_texture_latent_20260630/v293a_patch_view_moe_texture_latent_dim8_fixedviewsupport_20260630/v180_perceptual_surface_decoder_audit.json
outputs/carnet/spcarnet_v293_texture_latent_20260630/v293b_warmstart_texture_latent_dim4_fixedviewsupport_20260630/v180_perceptual_surface_decoder_audit.json
```

## Next Step

Do not run full9 yet. The next useful step is not another latent-dim scan. Keep
the texture-latent carrier, but add a target-blind perceptual/tail reliability
head or regularizer that suppresses the target views responsible for negative
SSIM/LPIPS tails while preserving v293a's PSNR gain.

```text
Final status: NOT COMPLETE.
```
