# v333 Target-Neighbor Consistency Certificate

Date: 2026-07-01

## Purpose

v331 showed that source-heldout pairwise LCB evidence is over-confident on
treehill's bad promoted target views. v332 showed that support-dropout
stability also cannot separate those bad views from positive controls. v333 adds
a different target-blind evidence family: target-neighborhood render
self-consistency.

The core question is whether a promoted candidate, when warped into nearby
target cameras using the already available target render/depth/camera, becomes
less consistent with neighboring base renders than its incumbent. The mechanism
does not use target/test GT for the decision. Target GT is read only after image
save for normal evaluation.

## Implemented Interfaces

Main apply pipeline:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New opt-in CLI:

```text
--enable_target_neighbor_consistency_certificate
--target_neighbor_consistency_mode {shadow,enforce}
--target_neighbor_consistency_sources pairwise
--target_neighbor_consistency_min_incumbent_minus_output_delta
--target_neighbor_consistency_neighbor_k
--target_neighbor_consistency_direction_weight
--target_neighbor_consistency_max_side
--target_neighbor_consistency_depth_abs_tol
--target_neighbor_consistency_depth_rel_tol
--target_neighbor_consistency_min_confidence
--target_neighbor_consistency_min_effective_weight
```

Diagnostic probe:

```text
scripts/car_model/probe_target_neighbor_self_consistency.py
```

Committed result artifacts:

```text
docs/car_model/results/v333_target_neighbor_consistency_probe_treehill_base_reference.json
docs/car_model/results/v333_target_neighbor_consistency_probe_treehill_same_variant.json
docs/car_model/results/v333_target_neighbor_consistency_shadow_treehill_report.json
docs/car_model/results/v333_target_neighbor_consistency_enforce_treehill_report.json
docs/car_model/results/v333_target_neighbor_consistency_enforce_stump_report.json
docs/car_model/results/v333_target_neighbor_consistency_full9_vs_v329b_audit.json
docs/car_model/results/v333_target_neighbor_consistency_full9_vs_v329b_audit.md
```

Full output dirs:

```text
outputs/carnet/spcarnet_v333_target_neighbor_treehill_20260701
outputs/carnet/spcarnet_v333_target_neighbor_consistency_shadow_treehill_20260701
outputs/carnet/spcarnet_v333_target_neighbor_consistency_enforce_treehill_20260701
outputs/carnet/spcarnet_v333_target_neighbor_consistency_enforce_stump_20260701
outputs/carnet/spcarnet_v333_target_neighbor_consistency_full9_20260701
```

W&B offline runs:

```text
outputs/carnet/spcarnet_v333_target_neighbor_consistency_enforce_treehill_20260701/wandb/offline-run-20260701_033928-te5cmewp
outputs/carnet/spcarnet_v333_target_neighbor_consistency_shadow_treehill_20260701/wandb/offline-run-20260701_034037-b1y647sl
outputs/carnet/spcarnet_v333_target_neighbor_consistency_enforce_stump_20260701/wandb/offline-run-20260701_034129-6jeou93q
```

## Mechanism

For a post-decision promotion from `incumbent_variant` to `output_variant`, v333:

1. selects the nearest target cameras by the existing camera-distance/direction
   support selector;
2. warps the current target candidate image into each neighboring target camera
   using target depth and camera geometry;
3. compares the warped candidate to the neighbor's base render with a
   confidence-weighted MAE;
4. repeats the same scoring for the incumbent;
5. computes `incumbent_minus_output_mae_delta`.

If the candidate is more inconsistent than the incumbent by more than the
configured margin, the certificate can shadow-log or enforce a rollback to the
incumbent. The focused v333 setting uses:

```text
mode: enforce for promoted result, shadow for ablation
sources: pairwise
threshold: -0.0001
neighbor_k: 2
max_side: 256
```

## Diagnostic Probe

The base-reference probe gave a useful but imperfect signal on the risky
treehill `mix0250` promotions:

| view | target PSNR delta vs fixed | target SSIM delta vs fixed | incumbent-minus-output consistency delta |
|---|---:|---:|---:|
| 00002 | +0.007259 | +0.000269 | +0.00029352 |
| 00004 | +0.002511 | -0.000220 | +0.00004925 |
| 00007 | -0.026469 | -0.000043 | -0.00012174 |
| 00008 | -0.004946 | -0.000321 | -0.00014801 |
| 00009 | -0.012385 | -0.000004 | -0.00002499 |
| 00011 | +0.033720 | +0.000269 | -0.00007005 |
| 00015 | +0.015077 | +0.000209 | -0.00004443 |

The same-variant probe was not useful as a veto because it was mostly positive
even on target-negative views. It is therefore retained as diagnostic evidence,
not promoted into the policy.

## Apply Results

| run | scene | mode | selected PSNR gain | selected SSIM gain | target-neighbor rollback | verdict |
|---|---|---|---:|---:|---:|---|
| v331 reference | treehill | no target-neighbor certificate | 0.104664074413 | 0.001673645443 | 0 | baseline for this probe |
| v333 shadow | treehill | shadow | 0.104664074413 | 0.001673645443 | 2 would rollback | ablation, no image change |
| v333 enforce | treehill | enforce | 0.106409362285 | 0.001693874598 | 2 applied | improves over v331 |
| v333 enforce | stump | enforce | 0.057029761393 | 0.001208242029 | 0 applied | no-harm sanity check |

Treehill improvement over v331:

```text
PSNR gain: +0.001745287872
SSIM gain: +0.000020229154
```

The enforced rollbacks are `00007` and `00008`. They are both true target
regressions under the post-save evaluation. Positive controls `00002`, `00011`,
and `00015` are kept.

## Full9 Replay

After the focused treehill/stump checks, v333 was replayed on the same full9
scene set used by v329b, with one frozen policy across scenes.

Audit files:

```text
docs/car_model/results/v333_target_neighbor_consistency_full9_vs_v329b_audit.json
docs/car_model/results/v333_target_neighbor_consistency_full9_vs_v329b_audit.md
```

Macro result:

| metric | v329b | v333 | delta |
|---|---:|---:|---:|
| selected PSNR gain | 0.272522652479 | 0.272716573354 | +0.000193920875 |
| selected SSIM gain | 0.003736660673 | 0.003738908357 | +0.000002247684 |
| target-neighbor rollback count | 0 | 2 | +2 |

Per-scene result:

| scene | PSNR gain delta vs v329b | SSIM gain delta vs v329b | rollbacks |
|---|---:|---:|---:|
| bicycle | +0.000000000000 | +0.000000000000 | 0 |
| flowers | +0.000000000000 | +0.000000000000 | 0 |
| garden | +0.000000000000 | +0.000000000000 | 0 |
| stump | +0.000000000000 | +0.000000000000 | 0 |
| counter | +0.000000000000 | +0.000000000000 | 0 |
| treehill | +0.001745287872 | +0.000020229154 | 2 |
| bonsai | +0.000000000000 | +0.000000000000 | 0 |
| room | +0.000000000000 | +0.000000000000 | 0 |
| kitchen | +0.000000000000 | +0.000000000000 | 0 |

Reading: v333 is full9-positive, but the full9 gain is narrow and entirely
comes from treehill tail repair. This is a valid milestone and a safer candidate
than v331/v332, but not a broad capability jump.

## Commands

Treehill enforce:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/treehill/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --output_dir outputs/carnet/spcarnet_v333_target_neighbor_consistency_enforce_treehill_20260701 \
  --policy_profile v322c_incumbent \
  --enable_pairwise_dominance_policy \
  --pairwise_dominance_enable_ood_guard \
  --pairwise_dominance_min_local_ssim_delta -0.001 \
  --pairwise_dominance_min_local_min_delta -0.005 \
  --pairwise_dominance_min_source_ssim_delta -0.0002 \
  --pairwise_dominance_min_source_min_delta -0.005 \
  --pairwise_dominance_max_blend_step 0.25 \
  --source_reliability_enable_fixed_rollback_certificate \
  --source_reliability_fixed_rollback_min_objective_margin 0.005 \
  --source_reliability_fixed_rollback_min_psnr_margin 0.005 \
  --source_reliability_fixed_rollback_min_ssim_margin 0.0 \
  --source_reliability_fixed_rollback_min_best_psnr_delta 0.005 \
  --source_reliability_fixed_rollback_min_best_ssim_delta 0.0 \
  --source_reliability_fixed_rollback_max_scene_opposition_fraction 0.05 \
  --source_reliability_fixed_rollback_min_scene_aligned_fraction 0.9 \
  --enable_promotion_rollback_certificate \
  --promotion_rollback_mode shadow \
  --promotion_rollback_min_lcb_psnr_delta 0.0 \
  --promotion_rollback_min_lcb_ssim_delta 0.0 \
  --promotion_rollback_min_local_cvar_delta 0.0 \
  --promotion_rollback_min_local_min_delta -0.005 \
  --promotion_rollback_max_local_negative_fraction 0.10 \
  --enable_target_neighbor_consistency_certificate \
  --target_neighbor_consistency_mode enforce \
  --target_neighbor_consistency_min_incumbent_minus_output_delta -0.0001 \
  --target_neighbor_consistency_neighbor_k 2 \
  --target_neighbor_consistency_max_side 256 \
  --copy_gt \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v333_target_neighbor_consistency_enforce_treehill
```

## Interpretation

v333 is the first post-v331/v332 attempt that converts the reflection into a
positive full9 pipeline result. It is more credible than another source-side
scalar gate because it tests target-camera geometry without using target GT.

However, it is not a final paper endpoint:

- it improves treehill and full9 over v329b, but the gain is still modest;
- `00009` remains a target-negative promotion that the current threshold keeps;
- the full9 gain is concentrated in one scene rather than broad across scenes;
- same-variant target-neighborhood consistency failed as a discriminator,
  confirming that model self-coherence alone is not enough.

## Verdict

Final status: NOT COMPLETE.

v333 should be kept as a real method improvement and as the current
post-v329b candidate, but it is not yet a 100% paper-level closed loop.
