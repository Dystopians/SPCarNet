# v177-v179 Full-Rank Texture Decoder Audit

Date: 2026-06-29

## Purpose

This log follows `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.
The v176 audit showed that the Phase-J teacher signal is real, but the existing
face/UV residual carrier improves residual MSE while hurting SSIM and LPIPS on
flowers policy-val. v177-v179 test whether the problem is specifically the
low-rank texture bottleneck, or whether the current baked residual projection is
misaligned with perceptual/structural quality.

## Implemented Method Change

`scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` now
supports full-rank teacher texture decoder modes:

- `full_rank_view_texture_rich`: stores the complete per-face/UV-bin rich-feature
  linear residual decoder without SVD compression. It uses the same 18-D feature
  vector as `low_rank_view_texture_rich`.
- `full_rank_surface_feature_rff_texture`: stores a complete per-bin 50-D decoder
  with deterministic UV Fourier/RFF features and normal-view interactions.

This is a real representation change in the train/eval pipeline. The fitted
decoder is used by the existing `fit_atlas -> evaluate_policy_val -> apply`
path, not by an external image post-process.

## Commands

v177 full-rank rich vs low-rank rich:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_phasej_teacher_residual_projection.py \
  --compute_lpips \
  --projection_modes none,low_rank_view_texture_rich,full_rank_view_texture_rich \
  --max_candidate_faces 512 \
  --max_candidate_face_samples_per_view 4096 \
  --max_samples_per_view 2048 \
  --max_policy_val_samples_per_view 2048 \
  --alpha_grid 0,0.0625,0.125,0.25,0.5 \
  --output_json /tmp/peilincai_spcarnet_v177_fullrank_projection_audit.json \
  --output_md /tmp/peilincai_spcarnet_v177_fullrank_projection_audit.md
```

v178 full-rank RFF texture:

```bash
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_phasej_teacher_residual_projection.py \
  --compute_lpips \
  --projection_modes full_rank_surface_feature_rff_texture \
  --max_candidate_faces 128 \
  --max_candidate_face_samples_per_view 4096 \
  --max_samples_per_view 2048 \
  --max_policy_val_samples_per_view 2048 \
  --alpha_grid 0,0.0625,0.125,0.25,0.5 \
  --output_json /tmp/peilincai_spcarnet_v178_rff_fullrank_projection_audit.json \
  --output_md /tmp/peilincai_spcarnet_v178_rff_fullrank_projection_audit.md
```

v179 full-rank RFF texture with edge/luma residual target:

```bash
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_phasej_teacher_residual_projection.py \
  --compute_lpips \
  --projection_modes full_rank_surface_feature_rff_texture \
  --teacher_residual_target_mode edge_luma_mix \
  --teacher_residual_target_luma_mix 0.5 \
  --teacher_residual_target_edge_boost 0.5 \
  --max_candidate_faces 128 \
  --max_candidate_face_samples_per_view 4096 \
  --max_samples_per_view 2048 \
  --max_policy_val_samples_per_view 2048 \
  --alpha_grid 0,0.0625,0.125,0.25,0.5 \
  --output_json /tmp/peilincai_spcarnet_v179_rff_edge_fullrank_projection_audit.json \
  --output_md /tmp/peilincai_spcarnet_v179_rff_edge_fullrank_projection_audit.md
```

## Shared Teacher Signal

On flowers train-policy-val:

| source | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| parent | 20.516130 | 0.729221 | 0.145680 |
| Phase-J teacher | 21.017230 | 0.745371 | 0.134851 |
| teacher gain | +0.501100 | +0.016150 | +0.010829 |

The teacher signal is valid. The projection failure is not caused by a zero or
miswired teacher.

## Projection Results

Best rows are selected by residual MSE gain over parent on policy-val.

| run | mode | pass all-axis | best alpha | MSE rel gain | SSIM gain | LPIPS gain | pos views | LPIPS pos views | decoder support |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v177 | none | False | 0.25 | 0.037612 | -0.000021 | -0.000038 | 0.583 | 0.333 | n/a |
| v177 | low_rank_view_texture_rich | False | 0.25 | 0.059065 | -0.000013 | -0.000020 | 0.667 | 0.333 | 132 faces, 4362 bins, rank 4 |
| v177 | full_rank_view_texture_rich | False | 0.25 | 0.061216 | -0.000015 | -0.000021 | 0.500 | 0.250 | 132 faces, 4362 bins, rank 18 |
| v178 | full_rank_surface_feature_rff_texture | False | 0.25 | 0.079371 | -0.000010 | -0.000015 | 0.667 | 0.333 | 43 faces, 1631 bins, rank 50 |
| v179 | full_rank_surface_feature_rff_texture + edge_luma_mix | False | 0.25 | 0.078375 | -0.000010 | -0.000016 | 0.667 | 0.333 | 43 faces, 1631 bins, rank 50 |

## Verdict

The new full-rank texture decoder increases residual MSE gain, especially with
RFF features, but it still does not improve SSIM or LPIPS on policy-val. Under
the v169 prompt, this blocks flowers exact and full9 promotion.

The important lesson is now sharper than v176:

- The failure is not merely an SVD rank-K bottleneck.
- The richer surface feature decoder can fit more teacher residual energy, but
  the additive baked residual still moves images in a direction that is not
  aligned with perceptual/structural metrics.
- Edge/luma target shaping does not fix this under the current projection path.

## Archived Artifacts

- `docs/car_model/vnext_artifacts/v177_fullrank_projection_audit.json`
- `docs/car_model/vnext_artifacts/v178_rff_fullrank_projection_audit.json`
- `docs/car_model/vnext_artifacts/v179_rff_edge_fullrank_projection_audit.json`

## Next Research Step

Do not continue with alpha scans, support expansion, or larger full-rank
surface residual decoders as the main route. The next plausible paper-level
route should change the optimization itself:

1. Train a small differentiable surface-feature residual decoder with an
   image-space objective that includes SSIM/LPIPS or edge-aware structure loss,
   rather than solving local RGB ridge regressions.
2. Keep strict train-fit/policy-val separation: train on train-fit teacher
   residual, select by policy-val GT, apply to target/test with GT stripped.
3. Use v176/v177-v179 as the entry gate: only run flowers exact if policy-val
   improves PSNR, SSIM, and LPIPS together.

Current status: NOT COMPLETE for paper readiness. COMPLETE only as a v169
diagnostic milestone.
