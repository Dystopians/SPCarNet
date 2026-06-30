# 2026-06-30 v296 Holdout Direction Surface Texture Scaffold

This note documents the first v296 method scaffold after the Phase-J stall
root-cause analysis.

Unlike v295, this is a real representation change: source-heldout residual
direction stability is now baked into the surface texture feature field used by
the neural decoder. It is still only smoke-validated here, so it is not a
quality claim.

## Motivation

v285/v286 showed that source-heldout residual direction calibration is useful
as a diagnostic, but in that line it only shrank target residuals during apply.
It did not give the decoder a stronger representation.

v294 then showed that the current carrier projection upper bound is far too
small. The next route needs to expose cross-view residual-direction stability to
the learned decoder itself.

## Implemented Change

Added a new surface texture mode:

```text
--surface_texture_mode lowrank_view_holdout_v3
```

This mode extends `lowrank_view_v2` with four train-fit, target-blind heldout
direction features per face/UV bin:

| feature | meaning |
|---|---|
| `holdout_cosine` | cosine between residual means from two source-view splits |
| `holdout_error_confidence` | `exp(-error_ratio)` style confidence from split disagreement |
| `holdout_support_balance` | balance between split support counts |
| `holdout_confidence` | combined direction, error, and support confidence |

Implementation summary:

- Source views are split by fit-view parity inside `_fit_surface_feature_texture`.
- The split statistics are computed from train-fit teacher residuals only.
- Both face fallback rows and face/UV-bin rows receive heldout features.
- `_surface_texture_reliability_from_rows` uses the v3 heldout confidence
  multiplied by the existing low-rank reliability.
- `lowrank_view_cos` view-support gate works with both `lowrank_view_v2` and
  `lowrank_view_holdout_v3`.
- The parser, checkpoint mode checks, NPZ save path, and decoder-output mode
  validation now accept v3.

## Smoke Validation

Static checks:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/train_perceptual_surface_residual_decoder.py
git diff --check -- scripts/car_model/train_perceptual_surface_residual_decoder.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py --help | rg "lowrank_view_holdout_v3|surface_texture_mode"
```

Smoke command:

```text
CUDA_VISIBLE_DEVICES=5 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_eval_mode never \
  --output_dir /tmp/peilincai_spcarnet_v296_holdout_v3_smoke_20260630 \
  --surface_texture_mode lowrank_view_holdout_v3 \
  --surface_texture_uv_bins 4 \
  --surface_texture_max_samples_per_view 10000 \
  --decoder_output_mode patch_view_moe \
  --max_candidate_faces 8 \
  --max_candidate_face_samples_per_view 256 \
  --steps 1 \
  --batch_size 512 \
  --policy_val_stride 12 \
  --alpha_grid 0,0.03125 \
  --eval_chunk_size 16384 \
  --policy_val_ssim_max_side 256
```

Smoke artifact root:

```text
/tmp/peilincai_spcarnet_v296_holdout_v3_smoke_20260630
```

Key smoke facts:

| field | value |
|---|---:|
| surface texture mode | `lowrank_view_holdout_v3` |
| feature dim | 47 |
| covered bin fraction | 0.445312 |
| mean lowrank reliability | 0.159655 |
| mean source camera concentration | 0.950959 |
| mean holdout cosine | 0.364887 |
| mean holdout error confidence | 0.539536 |
| mean holdout support balance | 0.533647 |
| mean holdout confidence | 0.288165 |
| policy-val all-axis pass | false |
| target exact ran | false |

The smoke did not show quality improvement: selected alpha was `0.0`. This is
expected for a 1-step, 8-face interface smoke and should not be interpreted as a
method result.

## Required Next Experiment

Run a fair same-budget reduced comparison:

1. `lowrank_view_v2` baseline, same faces/steps/policy-val split.
2. `lowrank_view_holdout_v3`, same command except surface texture mode.
3. Optional v3 with `--confidence_head --confidence_target_mode texture_direction`
   and mild confidence weighting, because v3 now exposes target-blind heldout
   confidence to the confidence head.

If the v3 reduced run improves policy-val all-axis without worse tails, then run
flowers exact with target no-GT apply. If not, do not promote it; move to a
stronger source-heldout auxiliary direction loss.

## Reduced Same-Budget Result

The fair reduced comparison has now completed with W&B offline logging:

```text
docs/car_model/results/v296_reduced_v2_v3_comparison_summary.json
/tmp/peilincai_spcarnet_v296_reduced_v2_20260630
/tmp/peilincai_spcarnet_v296_reduced_v3_20260630
```

| mode | feature dim | covered bins | holdout cosine | holdout confidence | selected alpha | policy-val all-axis |
|---|---:|---:|---:|---:|---:|---|
| `lowrank_view_v2` | 43 | 0.507812 | n/a | n/a | 0.000000 | false |
| `lowrank_view_holdout_v3` | 47 | 0.507812 | 0.356491 | 0.286819 | 0.000000 | false |

Nonzero alpha rows changed only about `4.6e-6` to `3.3e-5` of pixels and did
not pass all-axis policy-val. v3 did not beat v2 under this reduced budget.

Conclusion: v3 is a useful diagnostic/interface scaffold, but it is not a
quality improvement. The next method should not just expose heldout statistics
as features; it needs an explicit source-heldout residual transport loss.

## Status

Final status: NOT COMPLETE.

v296 is now implementable and smoke-tested, but it has not yet proven a
Phase-J-scale quality gain. The reduced comparison is negative, so flowers exact
and full9 promotion remain blocked for this route.
