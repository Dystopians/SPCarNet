# 2026-06-30 v295 Texture Reliability Interface Closure

This note documents an engineering interface closure for
`scripts/car_model/train_perceptual_surface_residual_decoder.py`.

It is not a Phase-J quality breakthrough. It fixes a half-wired texture-bin
reliability calibration path so future v295 experiments can be audited fairly.

## Motivation

After the Phase-J stall investigation, the next credible research route needs
stronger cross-view residual-direction modeling. Before that can be tested, the
current decoder script had a hygiene issue:

- texture-bin reliability calibration helpers existed;
- policy-val grids partly accepted texture thresholds;
- but target exact apply, `_predict_delta_image`, CLI flags, payload, W&B fields,
  and Markdown reporting were not consistently wired.

This could make a later experiment silently use one reliability policy on
policy-val and a different policy on target apply. That would be an unfair and
hard-to-debug protocol gap.

## Implemented Change

The script now supports texture-bin reliability as a first-class audited policy:

1. Added CLI flags:
   - `--enable_calibration_texture_reliability`
   - `--calibration_texture_min_bin_count`
   - `--texture_reliability_threshold`
   - `--texture_reliability_threshold_grid`

2. Added calibration split activation when either face or texture reliability is
   enabled.

3. Added texture-bin calibration payload and compressed artifact:
   - `calibration_texture_reliability.npz`
   - per-bin score, counts, mean L1 gain, mean structure gain, positive fraction.

4. Threaded selected texture reliability thresholds through:
   - policy-val grid;
   - best policy-val render writing;
   - `_predict_delta_image`;
   - target no-GT exact apply;
   - target exact per-view rows and top-level summary.

5. Added payload / W&B / Markdown reporting:
   - `calibration_texture_reliability`
   - `texture_reliability_threshold_grid`
   - `selected_texture_reliability_threshold`
   - `mean_texture_reliability_keep_fraction`
   - positive/valid texture-bin calibration fractions.

6. Fixed audit wording:
   - if target exact is not run, the interpretation now says it is a diagnostic
     only and cannot be used as a flowers exact or Phase-J comparison claim.

## Validation

Static checks:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/train_perceptual_surface_residual_decoder.py
git diff --check -- scripts/car_model/train_perceptual_surface_residual_decoder.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py --help | rg "texture_reliability|calibration_texture|face_reliability"
```

All passed.

Smoke command:

```text
CUDA_VISIBLE_DEVICES=5 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_eval_mode never \
  --output_dir /tmp/peilincai_spcarnet_v295_texture_reliability_smoke_20260630 \
  --surface_texture_mode lowrank_view_v2 \
  --surface_texture_uv_bins 4 \
  --surface_texture_max_samples_per_view 20000 \
  --decoder_output_mode patch_view_moe \
  --max_candidate_faces 16 \
  --max_candidate_face_samples_per_view 512 \
  --steps 1 \
  --batch_size 512 \
  --policy_val_stride 8 \
  --calibration_stride 5 \
  --enable_calibration_texture_reliability \
  --calibration_texture_min_bin_count 2 \
  --texture_reliability_threshold_grid=-1000000000,0 \
  --alpha_grid 0,0.03125 \
  --eval_chunk_size 16384 \
  --policy_val_ssim_max_side 256
```

Smoke artifact root:

```text
/tmp/peilincai_spcarnet_v295_texture_reliability_smoke_20260630
```

Key smoke facts:

| field | value |
|---|---:|
| texture calibration enabled | true |
| texture valid-bin fraction | 0.281250 |
| texture positive-bin fraction | 0.555556 |
| texture threshold grid | `[-1000000000.0, 0.0]` |
| selected texture threshold | -1000000000.0 |
| selected texture keep fraction | 1.000000 |
| target exact ran | false |
| target/test GT used for apply | false |
| no-target-GT audit pass | true |

The smoke was intentionally tiny:

- `steps=1`
- no LPIPS
- `target_eval_mode=never`
- only 16 candidate faces

Therefore it is not a quality result and must not be used as a Phase-J
comparison.

## Interpretation

This closes an interface gap. It does not solve the Phase-J bottleneck.

The next research step remains the one identified in:

```text
docs/car_model/6-30-PhaseJ-Stall-RootCause-Reflection.md
docs/car_model/6-30-v294-CrossViewResidualDirection-Synthesis.md
```

Specifically, the method needs a source-heldout cross-view residual direction
predictor or multi-source residual basis, not another scalar reliability gate.

## Status

Final status: NOT COMPLETE.

The repository is cleaner for future v295 experiments, but this is only an
engineering closure milestone.
