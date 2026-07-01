# 2026-06-30 v298 High-Bandwidth Source-Heldout ELA Diagnostic

This milestone adds a diagnostic tool for the post-Phase-J bottleneck:

```text
scripts/car_model/diagnose_source_heldout_ela_transport.py
```

It does not claim a new target/test method.  It asks a narrower and important
question:

```text
If we keep the Phase-J / ELA target-conditioned support-view warp path, does
train-only source-heldout residual transport still have useful headroom?
```

The answer on flowers is yes.

## Why This Diagnostic Matters

v294/v296/v297 showed that the current baked face/UV/bin/latent carrier is too
weak:

- v294 carrier upper-bound: only `+0.000164 dB` policy-val PSNR;
- v296 heldout features: selected `alpha=0.0`;
- v297 transport loss: real objective-level change, but first pilot remained
  numerical-noise scale and alpha expansion made PSNR more negative.

v298 tests the opposite side of the hypothesis.  It keeps the high-bandwidth ELA
information path and measures source-heldout repair on train views only.

## Implemented Tool

The new script:

1. loads a rendered train split with `renders/`, `gt/`, `depths/`, and
   `camera_index.json`;
2. splits train frames into source and heldout-source views;
3. computes ELA residual signal from source views only;
4. applies an alpha sweep to heldout-source views;
5. reports PSNR, SSIM, changed fraction, source-heldout direction cosine,
   residual energy ratio, W&B offline path, JSON, Markdown, and optional visual
   examples.

This diagnostic uses heldout train GT only for measurement.  It does not read
target/test GT.

## Validation

Static checks:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/diagnose_source_heldout_ela_transport.py
git diff --check -- scripts/car_model/diagnose_source_heldout_ela_transport.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/diagnose_source_heldout_ela_transport.py --help
```

2-view smoke:

```text
/tmp/peilincai_spcarnet_source_heldout_ela_smoke_20260630
```

Smoke result:

| field | value |
|---|---:|
| heldout views | 2 |
| best alpha | 0.25 |
| PSNR gain | +0.061288 |
| SSIM gain | +0.000684 |
| changed fraction | 0.693904 |
| all-axis pass | true |

## Full Heldout Flowers Diagnostic

Command:

```text
WANDB_MODE=offline \
WANDB_DIR=/tmp/peilincai_wandb_v298_source_heldout_ela_20260630 \
CUDA_VISIBLE_DEVICES=5 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/diagnose_source_heldout_ela_transport.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --output_dir outputs/carnet/spcarnet_v298_source_heldout_ela_transport_20260630 \
  --device cuda \
  --heldout_stride 4 \
  --heldout_offset 0 \
  --k 4 \
  --alpha_grid 0,0.125,0.25,0.5,0.75,1 \
  --evidence_max_side 256 \
  --compute_ssim \
  --ssim_max_side 256 \
  --save_example_views 2 \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v298-source-heldout-ela-flowers-fullheldout
```

Output:

```text
outputs/carnet/spcarnet_v298_source_heldout_ela_transport_20260630
```

W&B offline:

```text
outputs/carnet/spcarnet_v298_source_heldout_ela_transport_20260630/wandb/offline-run-20260630_170937-urlbgvz1
```

Split:

| field | value |
|---|---:|
| train views | 151 |
| source views | 113 |
| heldout views | 38 |
| heldout stride | 4 |
| heldout offset | 0 |

Main summary:

| field | value |
|---|---:|
| best alpha | 0.25 |
| best PSNR gain | +0.075520 |
| best SSIM gain | +0.001146 |
| best changed fraction | 0.683378 |
| mean direction cosine | 0.159510 |
| mean energy ratio L1 | 0.350456 |
| mean covered fraction | 0.912223 |
| all-axis pass | true |

Alpha sweep:

| alpha | PSNR gain | SSIM gain | changed | PSNR positive views | SSIM positive views | PSNR min | PSNR CVaR20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 | +0.000000 | +0.000000 |
| 0.125 | +0.044094 | +0.001014 | 0.486012 | 1.000000 | 0.868421 | +0.004617 | +0.023381 |
| 0.250 | +0.075520 | +0.001146 | 0.683378 | 0.973684 | 0.789474 | -0.001816 | +0.034667 |
| 0.500 | +0.099431 | -0.001080 | 0.794748 | 0.947368 | 0.447368 | -0.047391 | +0.016910 |
| 0.750 | +0.071092 | -0.006352 | 0.828342 | 0.789474 | 0.131579 | -0.134971 | -0.054045 |
| 1.000 | -0.008112 | -0.014298 | 0.843331 | 0.421053 | 0.000000 | -0.261655 | -0.179951 |

Visual examples:

```text
outputs/carnet/spcarnet_v298_source_heldout_ela_transport_20260630/visuals
```

## Interpretation

v298 is a strong diagnostic result, not a completed new paper endpoint.

It shows that the Phase-J style high-bandwidth support-view warp path still has
nontrivial train-only source-heldout headroom:

- changed fraction is two to four orders larger than v296/v297 baked-carrier
  pilots;
- best PSNR/SSIM gains are positive on the full source-heldout split;
- alpha `0.25` is much safer than alpha `0.5` or larger for SSIM/tail risk;
- the mean direction cosine is still low (`0.1595`), which explains why a
  distilled static carrier can fail even when the online ELA path succeeds.

The key lesson is:

```text
The next representation should not merely fit a face/UV residual texture.
It should distill the target-conditioned support-view transport path itself.
```

## Next Method Direction

The next implementation should turn v298 from a diagnostic into a train/eval
method:

1. cache source-view residual features and depth-consistency statistics;
2. expose target-conditioned support-warp features to the decoder;
3. train source-A to heldout-source-B with RGB residual, direction, magnitude,
   SSIM/patch, and uncertainty losses;
4. certify on policy-val with nontrivial changed fraction and tail-safe
   PSNR/SSIM/LPIPS;
5. only then run flowers target exact and full9.

## Final Status

```text
Final status: NOT COMPLETE.
```

v298 gives the clearest positive next-step evidence so far: Phase-J's
high-bandwidth path is worth distilling, while the previous baked carrier alone
is not enough.
