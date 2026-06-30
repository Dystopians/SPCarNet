# SPCarNet v256 Policy-Val L1 Reliability Log

Date: 2026-06-29

v256 is the first v253-family run that fixes the target-exact mean LPIPS
regression. It is still not paper-complete, but it is a real algorithmic step:
learn a target-blind reliability map from held-out policy-val evidence and then
apply the fixed map to stripped target no-GT evidence.

## Method

Implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

New policy options:

```text
--policy_reliability_mode local_l1
--policy_reliability_alpha
--policy_reliability_min_count
--policy_reliability_min_positive_fraction
--policy_reliability_min_mean_gain
--policy_reliability_gain_scale
--policy_reliability_floor
```

The method:

1. Load or build the v253 source-feature residual bank.
2. On train-policy-val only, render the bank residual at a calibration alpha.
3. For every face/UV bin, accumulate local parent-vs-candidate L1 improvement
   against policy-val GT.
4. Convert positive fraction and mean local gain into a reliability score.
5. Freeze the reliability map.
6. Apply to target using stripped no-GT evidence.
7. Load target GT only after apply for evaluation.

This is not target/test GT tuning. Target GT is not visible while constructing
the reliability map or rendering target predictions.

## Main Command Pattern

```bash
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=offline \
WANDB_DIR=/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256c_policy_l1_reliability_minpos048_targetexact/wandb \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_surface_deferred_source_residual_renderer.py \
  --bank_checkpoint /tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/v253_deferred_source_renderer_bank.npz \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --policy_reliability_mode local_l1 \
  --policy_reliability_alpha 0.03125 \
  --policy_reliability_min_count 8 \
  --policy_reliability_min_positive_fraction 0.48 \
  --policy_reliability_min_mean_gain 0.0 \
  --policy_reliability_gain_scale 0.00025 \
  --policy_reliability_floor 0.0 \
  --alpha_grid 0,0.00390625,0.0078125,0.015625,0.03125,0.046875,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5 \
  --compute_lpips \
  --target_eval_mode auto \
  --enable_wandb \
  --output_dir /tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256c_policy_l1_reliability_minpos048_targetexact
```

## Results

Phase-J flowers exact gate remains:

| metric | required relation | Phase-J |
|---|---:|---:|
| PSNR | higher | 20.304358 |
| SSIM | higher | 0.557770 |
| LPIPS | lower | 0.329222 |

v256 improves over the parent, but does not pass the Phase-J PSNR gate.

| run | min positive fraction | alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | target all-axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v256a | 0.52 | 0.125 | +0.002737 | +0.000087 | +0.000035 | +0.000830 | +0.000026 | +0.000013 | pass |
| v256b | 0.50 | 0.250 | +0.005508 | +0.000175 | +0.000070 | +0.001659 | +0.000050 | +0.000026 | pass |
| v256c | 0.48 | 0.500 | +0.010844 | +0.000343 | +0.000144 | +0.003185 | +0.000091 | +0.000050 | pass |

Target exact candidate metrics for v256c:

| PSNR | SSIM | LPIPS | changed fraction | LPIPS positive view fraction |
|---:|---:|---:|---:|---:|
| 19.835239 | 0.620001 | 0.180285 | 0.007788 | 0.818182 |

Tail caveat for v256c:

| target tail | value |
|---|---:|
| PSNR gain CVaR | +0.000313 |
| SSIM gain CVaR | -0.000005 |
| LPIPS gain CVaR | -0.000056 |

The mean target result is all-axis positive, but the target tails are not fully
safe yet.

## Artifact Index

- summary JSON:
  `docs/car_model/results/v256_policy_l1_reliability_summary.json`
- v256a audit:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256a_policy_l1_reliability_targetexact/v253_deferred_source_renderer_audit.json`
- v256a W&B:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256a_policy_l1_reliability_targetexact/wandb/offline-run-20260629_201415-tsv1tup6`
- v256b audit:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256b_policy_l1_reliability_minpos050_targetexact/v253_deferred_source_renderer_audit.json`
- v256b W&B:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256b_policy_l1_reliability_minpos050_targetexact/wandb/offline-run-20260629_201641-gfaajjfy`
- v256c audit:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256c_policy_l1_reliability_minpos048_targetexact/v253_deferred_source_renderer_audit.json`
- v256c W&B:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256c_policy_l1_reliability_minpos048_targetexact/wandb/offline-run-20260629_201901-7rm7opzk`
- v256c target exact renders:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256c_policy_l1_reliability_minpos048_targetexact/target_exact_fixed_policy`

## Interpretation

This is the strongest current result in the v253-v256 line:

- v253 introduced a real deferred source-feature representation but target LPIPS
  was slightly negative.
- v255 source-agreement confidence did not fix LPIPS.
- v256 policy-val local-L1 reliability makes policy-val tails all positive and
  target exact mean metrics all positive.

The limitation is still significant:

- v256c target exact PSNR is `19.835239`, still below the Phase-J flowers
  reference `20.304358`.
- Target SSIM/LPIPS tails remain slightly negative.
- Visual changed fraction is still under `0.008`, so qualitative improvements
  are likely subtle.

Do not launch full9 from v256 yet. The next step should either strengthen the
teacher residual carrier or add a stronger tail-safe reliability objective that
uses policy-val patch/gradient/perceptual proxies instead of local L1 only.

Final status: **NOT COMPLETE for paper-level all-axis win**.
