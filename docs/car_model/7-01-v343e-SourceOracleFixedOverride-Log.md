# v343e Source-Oracle Fixed Override Log

Run stamp: 20260701

This milestone is a targeted fix after the v340d-v342e diagnostics showed a
specific failure mode: a source-heldout oracle could identify `fixed` as the
best local repair candidate, but target-neighbor consistency and the later
candidate-unlock stage could still erase that decision before final output.

## Method Change

Implemented in:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New mechanism:

```text
--target_neighbor_consistency_enable_source_oracle_fixed_override
```

The target-neighbor consistency certificate now receives
`source_oracle_knn_diagnostics`. When the decision source is
`source_oracle_knn`, the proposed output is `fixed`, and the scene incumbent is
not `fixed`, the policy can keep the fixed output only if both conditions hold:

1. source-heldout local evidence says fixed is a strong candidate;
2. source-reliability agreement independently supports that fixed decision.

The accepted override also blocks the later target-neighbor candidate unlock
from overwriting the fixed output. This is important because the first
implementation could accept the override and still lose it to the next policy
stage.

Thresholds used in v343e:

```text
--target_neighbor_consistency_source_oracle_fixed_min_source_psnr_delta 0.02
--target_neighbor_consistency_source_oracle_fixed_min_source_ssim_delta 0.00005
--target_neighbor_consistency_source_oracle_fixed_min_source_cvar_delta 0.01
--target_neighbor_consistency_source_oracle_fixed_min_source_min_delta 0.01
--target_neighbor_consistency_source_oracle_fixed_min_reliability_psnr_delta 0.03
--target_neighbor_consistency_source_oracle_fixed_min_reliability_ssim_delta 0.0001
```

## Validation Commands

Representative scene command:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<gpu> PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
--base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
--base_method_name ours_26000_phasef_extra_compact_base \
--checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
--output_dir outputs/carnet/spcarnet_v343e_source_oracle_fixed_override_probe_20260701/<scene> \
--policy_profile v322c_incumbent \
--enable_adaptive_residual_candidate \
--enable_source_trust_residual_candidate \
--no-generated_candidate_disable_when_scene_fixed \
--generated_candidate_require_source_summary_safe \
--generated_candidate_min_source_summary_psnr_delta_vs_scene -0.0005 \
--generated_candidate_min_source_summary_ssim_delta_vs_scene -0.0001 \
--enable_fixed_scene_generated_source_summary_unlock \
--fixed_scene_generated_unlock_candidate_names adaptive,source_trust \
--fixed_scene_generated_unlock_min_source_psnr_delta 0.0005 \
--fixed_scene_generated_unlock_min_source_ssim_delta -0.0001 \
--fixed_scene_generated_unlock_freeze_incumbent \
--source_reliability_enable_fixed_rollback_certificate \
--source_reliability_fixed_rollback_min_objective_margin 0.005 \
--source_reliability_fixed_rollback_min_psnr_margin 0.005 \
--source_reliability_fixed_rollback_min_ssim_margin 0.0 \
--source_reliability_fixed_rollback_min_best_psnr_delta 0.005 \
--source_reliability_fixed_rollback_min_best_ssim_delta 0.0 \
--source_reliability_fixed_rollback_max_scene_opposition_fraction 0.05 \
--source_reliability_fixed_rollback_min_scene_aligned_fraction 0.9 \
--enable_pairwise_dominance_policy \
--enable_source_oracle_knn_policy \
--source_oracle_knn_apply_mode post_reliability_scene_only \
--source_oracle_knn_require_reliability_agreement \
--enable_target_neighbor_consistency_certificate \
--target_neighbor_consistency_mode enforce \
--target_neighbor_consistency_sources pairwise,source_oracle_knn \
--target_neighbor_consistency_enable_source_contradiction \
--target_neighbor_consistency_contradiction_min_source_local_min_delta 0.01 \
--target_neighbor_consistency_contradiction_min_source_local_cvar_delta 0.01 \
--target_neighbor_consistency_contradiction_min_source_positive_fraction 1.0 \
--target_neighbor_consistency_contradiction_max_incumbent_minus_output_delta -0.00002 \
--target_neighbor_consistency_enable_source_oracle_fixed_override \
--target_neighbor_consistency_source_oracle_fixed_min_source_psnr_delta 0.02 \
--target_neighbor_consistency_source_oracle_fixed_min_source_ssim_delta 0.00005 \
--target_neighbor_consistency_source_oracle_fixed_min_source_cvar_delta 0.01 \
--target_neighbor_consistency_source_oracle_fixed_min_source_min_delta 0.01 \
--target_neighbor_consistency_source_oracle_fixed_min_source_positive_fraction_delta 0.0 \
--target_neighbor_consistency_source_oracle_fixed_min_reliability_psnr_delta 0.03 \
--target_neighbor_consistency_source_oracle_fixed_min_reliability_ssim_delta 0.0001 \
--enable_target_neighbor_candidate_unlock \
--enable_target_neighbor_all_candidate_diagnostic \
--copy_gt --enable_wandb
```

Static checks:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/apply_source_heldout_support_transport_calibrator.py
git diff --check -- scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

## Artifacts

```text
outputs/carnet/spcarnet_v343e_source_oracle_fixed_override_probe_20260701/
docs/car_model/results/v343e_source_oracle_fixed_override_focus6_oracle_gap.json
docs/car_model/results/v343e_source_oracle_fixed_override_focus6_oracle_gap.md
```

Each scene run used offline W&B logging under the corresponding output
directory.

## Quantitative Result

Focus6 scenes: bicycle, bonsai, kitchen, room, stump, treehill.

| scene | v342e PSNR gain | v342e SSIM gain | v343e PSNR gain | v343e SSIM gain | delta PSNR | delta SSIM |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 0.119958549 | 0.002988751 | 0.119958549 | 0.002988751 | +0.000000000 | +0.000000000 |
| bonsai | 0.575974442 | 0.005847958 | 0.582901932 | 0.005913528 | +0.006927490 | +0.000065570 |
| kitchen | 0.493623161 | 0.003910862 | 0.493623161 | 0.003910862 | +0.000000000 | +0.000000000 |
| room | 0.450326866 | 0.005142007 | 0.453250186 | 0.005189245 | +0.002923320 | +0.000047237 |
| stump | 0.058909355 | 0.001223635 | 0.058909355 | 0.001223635 | +0.000000000 | +0.000000000 |
| treehill | 0.118121383 | 0.001717435 | 0.116350575 | 0.001734800 | -0.001770808 | +0.000017365 |

Macro result:

| method | macro PSNR gain | macro SSIM gain | oracle PSNR gain | PSNR headroom |
|---|---:|---:|---:|---:|
| v342e | 0.302818959 | 0.003471775 | 0.314581083 | +0.011762124 |
| v343e | 0.304165626 | 0.003493470 | 0.314581083 | +0.010415457 |

Net effect over v342e:

```text
macro PSNR gain: +0.001346667
macro SSIM gain: +0.000021695
oracle headroom reduction: 0.011762124 -> 0.010415457
```

## What Actually Fired

The new fixed override accepted one view:

```text
scene: bonsai
view: 00035
output: fixed
source_psnr_delta: +0.042872629
source_ssim_delta: +0.000606620
source_cvar_delta: +0.014925405
source_min_delta: +0.014925405
reliability_predicted_psnr_delta: +0.269160545
reliability_predicted_ssim_delta: +0.002122137
```

This view was the largest v342e miss, so fixing it produced a measurable scene
and macro improvement.

## Negative Probe

v343f tried to restore treehill with pairwise OOD guarding:

```text
outputs/carnet/spcarnet_v343f_source_oracle_fixed_override_pairwiseguard_probe_20260701/
```

Result:

| scene | v343e PSNR gain | v343f PSNR gain | verdict |
|---|---:|---:|---|
| bonsai | 0.582901932 | 0.582901932 | unchanged |
| room | 0.453250186 | 0.450326866 | worse |
| treehill | 0.116350575 | 0.116350575 | unchanged |

This rules out a simple pairwise-guard flag fix.

## Reflection

The reflection was useful because it shifted the work away from blind parameter
scans and toward a concrete chain-of-responsibility bug:

```text
source oracle says fixed is best -> target consistency/target unlock can erase it
```

v343e repairs that chain and improves the macro result. However, the reflection
is not yet sufficient for a final paper-level method. The remaining weakness is
that the source-side fixed rollback machinery helps bonsai but changes the
treehill decision topology, removing earlier `mix0250` pairwise outputs and
causing a PSNR regression.

Next required fix:

```text
decouple target-time fixed rollback certificates from source-side LOO/pairwise
policy fitting, so the method can keep bonsai's fixed rescue without destroying
treehill's pairwise mixture evidence.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```
