# 2026-07-01 v342e Fixed-Scene Generated Unlock Freeze Log

## Verdict

`v342e_fixed_generated_unlock_freeze` is a small but real post-reflection
method update. It improves the focus6 macro result over `v341_source_trust` and
reduces oracle-selection headroom without using target GT for selection.

It is not a paper-level closure. The gain is still incremental, and the largest
misses are still fixed/learned arbitration failures.

| metric | v340d | v341 | v342e | v342e-v341 |
|---|---:|---:|---:|---:|
| macro selected PSNR gain | 0.301510278 | 0.302505694 | 0.302818959 | +0.000313266 |
| macro selected SSIM gain | 0.003461710 | 0.003469209 | 0.003471775 | +0.000002566 |
| macro oracle headroom | 0.012505689 | 0.012075390 | 0.011762124 | -0.000313266 |
| positive PSNR-headroom views | 65 | 68 | 74 | +6 |

Scene-level comparison:

| scene | v340d PSNR gain | v341 PSNR gain | v342e PSNR gain | v342e-v341 PSNR | v342e-v341 SSIM |
|---|---:|---:|---:|---:|---:|
| bicycle | 0.119958549 | 0.119958549 | 0.119958549 | +0.000000000 | +0.000000000 |
| bonsai | 0.576081269 | 0.575974442 | 0.575974442 | +0.000000000 | +0.000000000 |
| kitchen | 0.493623161 | 0.493623161 | 0.493623161 | +0.000000000 | +0.000000000 |
| room | 0.444247549 | 0.450326866 | 0.450326866 | +0.000000000 | +0.000000000 |
| stump | 0.057029761 | 0.057029761 | 0.058909355 | +0.001879594 | +0.000015393 |
| treehill | 0.118121383 | 0.118121383 | 0.118121383 | +0.000000000 | +0.000000000 |

The useful effect is concentrated on `stump`: the previous safe policy selected
`fixed` for all target views, while v342e admits the `adaptive` generated
candidate from source-heldout evidence and freezes it as the target-time
incumbent. This produces an all-axis improvement on `stump` while leaving
`treehill` unchanged.

## Implemented Method Change

File:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New opt-in controls:

```text
--enable_fixed_scene_generated_source_summary_unlock
--fixed_scene_generated_unlock_candidate_names
--fixed_scene_generated_unlock_min_source_psnr_delta
--fixed_scene_generated_unlock_min_source_ssim_delta
--fixed_scene_generated_unlock_min_source_cvar_delta
--fixed_scene_generated_unlock_min_source_min_delta
--fixed_scene_generated_unlock_ssim_weight
--fixed_scene_generated_unlock_freeze_incumbent
```

Mechanism:

- Run only after the source-heldout scene selector has selected `fixed`.
- Consider only active generated candidates, usually `adaptive` and
  `source_trust`.
- Compare each generated candidate against `fixed` on source-heldout summary
  statistics, not target GT.
- Require source PSNR gain and bounded source SSIM/tail risk before unlocking.
- If a candidate passes, replace the scene incumbent with that candidate.
- With `--fixed_scene_generated_unlock_freeze_incumbent`, keep that unlocked
  incumbent at target time so later aggressive per-view policies cannot promote
  it into a worse candidate.

This is a direct response to the v341/v342 negative probes:

- simply unsuppressing generated candidates hurt `treehill`;
- the old per-view risk model broke `stump`;
- global pairwise gate relaxation did not move `stump`.

The new policy is therefore intentionally narrow: it only changes fixed-scene
cases where source-heldout evidence already says a generated candidate is safer
than fixed.

## Commands

Primary focus6 root:

```text
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701
```

Representative command shape:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<gpu> PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
--base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
--base_method_name ours_26000_phasef_extra_compact_base \
--checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
--output_dir outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/<scene> \
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
--enable_pairwise_dominance_policy \
--enable_source_oracle_knn_policy \
--source_oracle_knn_require_reliability_agreement \
--enable_target_neighbor_consistency_certificate \
--enable_target_neighbor_candidate_unlock \
--enable_target_neighbor_all_candidate_diagnostic \
--copy_gt --enable_wandb
```

Analysis command:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_support_transport_oracle_gap.py \
--method v340d=outputs/carnet/spcarnet_v340d_source_oracle_agreement_pairwise_focus6_20260701 \
--method v341_source_trust=outputs/carnet/spcarnet_v341_source_trust_focus4_20260701 \
--method v342e_fixed_generated_unlock_freeze=outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701 \
--primary_metric psnr_gain \
--output_json docs/car_model/results/v342e_fixed_generated_unlock_freeze_focus6_oracle_gap.json \
--output_md docs/car_model/results/v342e_fixed_generated_unlock_freeze_focus6_oracle_gap.md
```

## Artifacts

Metrics:

```text
docs/car_model/results/v342e_fixed_generated_unlock_freeze_focus6_oracle_gap.json
docs/car_model/results/v342e_fixed_generated_unlock_freeze_focus6_oracle_gap.md
```

Per-scene reports:

```text
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/bicycle/support_transport_apply_report.json
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/bonsai/support_transport_apply_report.json
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/kitchen/support_transport_apply_report.json
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/room/support_transport_apply_report.json
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/stump/support_transport_apply_report.json
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/treehill/support_transport_apply_report.json
```

Offline W&B runs:

```text
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/bicycle/wandb/offline-run-20260701_083228-accrdamv
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/bonsai/wandb/offline-run-20260701_083331-h6ikjadh
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/kitchen/wandb/offline-run-20260701_083341-dhllm79w
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/room/wandb/offline-run-20260701_083351-na9pi8ma
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/stump/wandb/offline-run-20260701_083003-bqhthj3s
outputs/carnet/spcarnet_v342e_fixed_generated_unlock_freeze_probe_20260701/treehill/wandb/offline-run-20260701_083009-h1uvfebw
```

## Reflection: Did It Work?

The reflection did work, but only partially.

What improved:

- We stopped treating the fixed-scene failure as a generic hyperparameter
  problem.
- Negative probes isolated why broad relaxations are unsafe.
- The new policy is target-GT-free and only changes scenes where source-heldout
  evidence supports the generated candidate.
- `stump` now gets a real all-axis gain without sacrificing `treehill`.

What did not improve enough:

- The macro gain over v341 is only `+0.000313266` PSNR.
- The main oracle headroom is still `+0.011762124`.
- The largest misses still include `bonsai/00035`, `treehill/00011`,
  `treehill/00016`, `room/00023`, `kitchen/00018`, and multiple `stump` views.
- v342e solves one conservative-selection failure mode; it does not yet solve
  the core representation/selection gap.

Current conclusion:

```text
Reflection is useful and has now produced measurable progress, but it has not
yet produced a strong enough method for a paper-level closed loop.
```

## Next Required Work

The next route should not be more global threshold scanning. The data says the
remaining gap is view-local candidate arbitration:

1. Build a target-GT-free source-heldout local classifier for fixed-vs-learned
   reversals, trained/evaluated on source-heldout per-view examples.
2. Add explicit contradiction features for cases like `bonsai/00035`, where the
   current source-oracle KNN path selects learned but the target oracle prefers
   fixed.
3. Add a conservative promotion path for `treehill/stump` learned-best views
   that uses local support/tail evidence rather than scene-level fixed fallback.
4. Validate on focus6 first, then rerun the wider baseline/current/improved/
   ablation comparison before making paper claims.

Final status: NOT COMPLETE.
