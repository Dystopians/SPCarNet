# v209-v215 Low-Rank Surface Residual Gate Report

Date: 2026-06-29

## Verdict

Status: `NOT COMPLETE`.

The v169 prompt was followed in the important sense: before any full9 promotion, flowers was used as the decisive gate. We implemented a real representation change, ran policy-val projection checks, added a strict no-target-GT target apply path, and ran flowers exact apply. The method improved over the previous static carrier, but it still fails the Phase-J flowers gate because PSNR remains below the required `20.304358`.

## Method Change

New implementation:

- `scripts/car_model/train_surface_uv_residual_texture.py`
  - v209 static per-face UV teacher-residual micro-texture.
- `scripts/car_model/train_surface_lowrank_residual_texture.py`
  - v212+ low-rank view-conditioned surface residual texture.
  - Residual is represented as `sum_k weight_k(view, uv, parent) * basis_k(face, uv)`.
  - Supports `dir_uv_v1` and `dir_uv_parent_v1` basis modes.
  - Adds tail-safe promotion: by default, `best_all_axis` requires PSNR, SSIM, and LPIPS positive-view fractions to be `1.0`.
- `scripts/car_model/apply_surface_lowrank_residual_texture.py`
  - Strict no-GT target apply/eval interface.
  - Preflights target evidence keys before rendering.
  - Reads GT only after writing candidate renders for eval.

## Main Policy-Val Results

All rows use flowers teacher-surface evidence from:

`/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`

| run | carrier | faces | coverage | alpha | PSNR gain | SSIM gain | LPIPS gain | full cosine | full retention | active cosine | active retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v209 | static UV grid4 | 131072 | 0.794544 | 1.0 | +0.048768 | +0.001577 | +0.001115 | 0.218122 | 0.114063 | 0.298419 | 0.217277 |
| v210 | static UV grid4, higher coverage | 221673 | 0.900001 | 1.0 | +0.054650 | +0.001904 | +0.001242 | 0.231817 | 0.125084 | 0.290767 | 0.200060 |
| v211 | static UV grid8 | 131072 | 0.793459 | 1.0 | +0.047330 | +0.001898 | +0.001241 | 0.221262 | 0.109514 | 0.302834 | 0.208445 |
| v212 | low-rank dir_uv_v1 | 131072 | 0.794918 | 1.0 | +0.058915 | +0.002054 | +0.001340 | 0.248183 | 0.125387 | 0.339906 | 0.239390 |
| v213 | low-rank dir_uv_v1, higher coverage | 222308 | 0.900001 | 1.0 | +0.065698 | +0.002414 | +0.001463 | 0.261605 | 0.137613 | 0.327632 | 0.219595 |
| v214 | low-rank dir_uv_parent_v1 | 131072 | 0.794067 | 1.0 | +0.056282 | +0.001920 | +0.001210 | 0.244347 | 0.110494 | 0.334343 | 0.210403 |
| v215 | low-rank dir_uv_v1, 0.97 coverage, alpha grid to 2 | 524288 cap | 0.970 target | 1.5 tail-safe / 2.0 mean-best | +0.088619 / +0.090301 | +0.003323 / +0.003282 | +0.002606 / +0.003456 | 0.276123 | 0.339975 / 0.604399 | 0.322510 | 0.470834 / 0.837044 |

Key change: v215 proves the earlier representation was energy-limited. Full residual retention rose from v213 `0.137613` to v215 `0.604399` at alpha 2.0. However, high residual strength does not transfer safely to target.

## Flowers Exact No-GT Apply

Target no-GT evidence:

`/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`

Eval GT evidence, read only after render:

`/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`

| candidate | alpha policy | no-GT preflight | changed fraction | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS | Phase-J flowers gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| v213 exact | 1.0 | pass | 0.251365 | 19.877306 | 0.620489 | 0.180237 | +0.045252 | +0.000578 | +0.000098 | fail, PSNR below 20.304358 |
| v215 exact | 2.0 mean-best | pass | 0.357511 | 19.848502 | 0.617056 | 0.179364 | +0.016448 | -0.002854 | +0.000970 | fail, PSNR and SSIM |
| v215 exact | 1.5 tail-safe | pass | 0.354443 | 19.876960 | 0.619377 | 0.179763 | +0.044906 | -0.000533 | +0.000572 | fail, PSNR and SSIM |
| v215 exact | 1.0 conservative | pass | 0.346091 | 19.883736 | 0.620716 | 0.180202 | +0.051682 | +0.000806 | +0.000133 | fail, PSNR below 20.304358 |

Best exact row among these is v215 alpha 1.0 by PSNR/SSIM, but it is still far below the Phase-J flowers PSNR target.

## Commands

Representative commands:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/dev/shm WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_lowrank_residual_texture.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --policy_val_stride 4 --residual_rgb_key teacher_residual_rgb --residual_l1_key teacher_residual_l1 \
  --min_l1 0.0005 --min_alpha 0.03 --max_candidate_faces 524288 \
  --candidate_target_energy_coverage 0.97 --grid 4 --basis_mode dir_uv_v1 \
  --ridge_count 8 --min_bin_count 2 --alpha_grid 0,0.03125,0.0625,0.125,0.25,0.5,0.75,1,1.25,1.5,2 \
  --compute_lpips --output_dir /dev/shm/peilincai_spcarnet_v215_lowrank_grid4_diruv_524k_cov97_alpha2 \
  --enable_wandb --wandb_run_name v215-lowrank-grid4-diruv-524k-cov97-alpha2 --seed 215

CUDA_VISIBLE_DEVICES=2 TMPDIR=/dev/shm WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/apply_surface_lowrank_residual_texture.py \
  --checkpoint /dev/shm/peilincai_spcarnet_v215_lowrank_grid4_diruv_524k_cov97_alpha2/v212_lowrank_uv_residual_texture.npz \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --eval_gt_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --output_dir /dev/shm/peilincai_spcarnet_v215_lowrank_grid4_diruv_524k_cov97_alpha2/flowers_exact_apply_alpha1 \
  --alpha 1.0 --compute_lpips --enable_wandb --wandb_run_name v215-lowrank-flowers-exact-apply-alpha1
```

## Artifacts

- v213 policy-val JSON: `/dev/shm/peilincai_spcarnet_v213_lowrank_grid4_diruv_262k/v212_lowrank_uv_residual_texture_audit.json`
- v213 exact JSON: `/dev/shm/peilincai_spcarnet_v213_lowrank_grid4_diruv_262k/flowers_exact_apply/lowrank_target_apply_audit.json`
- v215 policy-val JSON: `/dev/shm/peilincai_spcarnet_v215_lowrank_grid4_diruv_524k_cov97_alpha2/v212_lowrank_uv_residual_texture_audit.json`
- v215 alpha1 exact JSON: `/dev/shm/peilincai_spcarnet_v215_lowrank_grid4_diruv_524k_cov97_alpha2/flowers_exact_apply_alpha1/lowrank_target_apply_audit.json`
- v215 alpha1.5 exact JSON: `/dev/shm/peilincai_spcarnet_v215_lowrank_grid4_diruv_524k_cov97_alpha2/flowers_exact_apply_alpha1p5/lowrank_target_apply_audit.json`
- v215 alpha2 exact JSON: `/dev/shm/peilincai_spcarnet_v215_lowrank_grid4_diruv_524k_cov97_alpha2/flowers_exact_apply_alpha2/lowrank_target_apply_audit.json`

## Lessons

1. Static UV residual texture was not enough. It improved all policy-val image metrics but had weak full projection.
2. Low-rank view-conditioned surface residual is a real improvement. v213 beats static carriers on policy-val and crosses full projection cosine 0.25.
3. Coverage and alpha solve retention, not transfer. v215 reaches full retention 0.604399 but exact target PSNR/SSIM does not improve enough.
4. Mean-best alpha is unsafe. v215 alpha 2.0 is policy-val mean-best but exact SSIM regresses; tail-safe alpha selection is now required.
5. The next bottleneck is target transfer/OOT support, not basic carrier capacity. The carrier can now hold teacher residual energy, but it applies that energy to target views too broadly or in wrong local regions.

## Next Step

Do not run full9. The next real method change should add target-support-aware confidence to the low-rank carrier:

- estimate support/OOT per target pixel from train evidence face/bin/view coverage;
- downweight high-coverage residual only where target view direction and local surface statistics are in-distribution;
- keep tail-safe policy selection;
- rerun flowers exact only after target exact PSNR moves near or above Phase-J.
