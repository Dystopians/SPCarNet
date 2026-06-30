# SPCarNet v253-v254 Deferred Source Renderer Log

Date: 2026-06-29

This log records the next execution step after
`docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md` and the v249-v252
representation-gate failures.

## Why This Is A Real Method Change

v249-v252 showed that static baked RGB carriers either damage SSIM/LPIPS tails or
collapse to near no-op. v253 therefore changes the representation:

- keep MeshSplatting parent geometry and parent RGB render;
- fit a train-view source bank from Phase-J teacher residual evidence;
- store multiple source residual entries per face/UV bin;
- at render time, aggregate source residuals by target view direction, normal
  agreement, parent RGB similarity, source support, and teacher gain;
- certify on train-policy-val before target/test evaluation;
- require stripped target no-GT evidence for apply.

Implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

The script writes JSON, Markdown, checkpointed source banks, policy-val renders,
target no-GT previews, and fixed-policy target exact render/parent/GT triplets.
It also supports `--bank_checkpoint`, so later policy/eval ablations can reuse a
frozen representation without rebuilding the bank or touching target GT.

## Commands

Main v253b run:

```bash
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=offline \
WANDB_DIR=/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/wandb \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_surface_deferred_source_residual_renderer.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --policy_val_stride 4 \
  --residual_rgb_key teacher_residual_rgb \
  --residual_l1_key teacher_residual_l1 \
  --max_candidate_faces 8192 \
  --candidate_target_energy_coverage 0.95 \
  --grid 4 \
  --source_top_k 6 \
  --compute_lpips \
  --target_eval_mode auto \
  --enable_wandb \
  --output_dir /tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact
```

Frozen-bank follow-up ablations:

```bash
--bank_checkpoint /tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/v253_deferred_source_renderer_bank.npz
```

were used for v253c, v253d, v254a, and v254b. These runs did not rebuild the
representation and did not use target GT for policy selection.

## Results

Phase-J flowers exact gate from v169:

| metric | required relation | Phase-J |
|---|---:|---:|
| PSNR | higher | 20.304358 |
| SSIM | higher | 0.557770 |
| LPIPS | lower | 0.329222 |

v253-v254 fixed-policy target exact results:

| run | residual mode | selected alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | target all-axis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v253b | raw RGB | 0.031250 | +0.001240 | +0.000015 | +0.000004 | +0.001063 | +0.000028 | -0.000002 | fail |
| v253c | raw RGB, fine alpha | 0.046875 | +0.001837 | +0.000020 | +0.000001 | +0.001579 | +0.000040 | -0.000007 | fail |
| v253d | raw RGB, conservative alpha | 0.015625 | +0.000628 | +0.000008 | +0.000006 | +0.000537 | +0.000014 | -0.000001 | fail |
| v254a | luma only | 0.031250 | +0.001141 | +0.000012 | +0.000002 | +0.000985 | +0.000025 | -0.000005 | fail |
| v254b | chroma shrink 0.25 | 0.031250 | +0.001166 | +0.000013 | +0.000003 | +0.001005 | +0.000025 | -0.000004 | fail |

The best target PSNR/SSIM movement is v253c, but it worsens LPIPS more. The
safest target LPIPS is v253d, but it still remains slightly negative and the
visual change is nearly no-op (`changed_fraction` about `0.000359`).

## No-GT Audit

The target apply evidence was:

```text
/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt
```

The no-GT audit passed: the target apply NPZs did not contain `rgb_gt`,
`residual_rgb`, `teacher_residual_rgb`, `teacher_residual_l1`, or teacher
benefit keys. Target GT was loaded only after no-GT apply from:

```text
/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented
```

## Artifact Index

- summary JSON:
  `docs/car_model/results/v253_v254_deferred_source_renderer_summary.json`
- v253b audit:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/v253_deferred_source_renderer_audit.json`
- v253b report:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/v253_deferred_source_renderer_audit.md`
- v253b target renders:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/target_exact_fixed_policy`
- v253b W&B offline:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/wandb/offline-run-20260629_194753-kac8i39s`
- v253c W&B offline:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253c_loadedbank_finealpha_targetexact/wandb/offline-run-20260629_195325-80t6pncf`
- v253d W&B offline:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253d_loadedbank_conservative_alpha_targetexact/wandb/offline-run-20260629_195533-i4rnd0z2`
- v254a W&B offline:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v254a_loadedbank_lumaonly_targetexact/wandb/offline-run-20260629_195825-facgdcze`
- v254b W&B offline:
  `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v254b_loadedbank_chromashrink_targetexact/wandb/offline-run-20260629_200035-ncgx2ahm`

## Interpretation

This is a genuine representation-level milestone: the v253 renderer is no longer
a static face atlas, alpha-only scan, or support-footprint tweak. It makes the
student representation view dependent and source-conditioned.

However, the result is still not paper-ready. The evidence is clear:

1. Policy-val all-axis is finally positive, but only by tiny margins.
2. Target exact PSNR and SSIM improve consistently.
3. Target exact LPIPS remains slightly negative for every fixed-policy variant.
4. Luma-only and chroma-shrink residual shaping do not fix the perceptual
   transfer issue.
5. The current source bank still carries too little reliable perceptual teacher
   signal: selected-alpha active projection retention is about `0.00119` of the
   teacher residual energy.

## Next Step

Do not run full9 from v253/v254. The next useful step is not another alpha grid
or channel transform. It should add a target-blind learned/certified confidence
term for each source/bin, trained to predict policy-val perceptual correctness
from source agreement, residual variance, view support, edge support, and
teacher-gain consistency. The renderer should then suppress bins whose
multi-source agreement is weak before applying residuals to target.

Final status: **NOT COMPLETE for paper-level all-axis win**.
