# 2026-07-01 v341 Source-Trust Residual Candidate Log

## Verdict

`v341_source_trust` is a real candidate-generation change and gives a small but
clean focus6 gain over `v340d`, but it is not a paper-level closure.

Compared with `v340d_source_oracle_agreement_pairwise_focus6_20260701` on the
same six scenes:

| metric | v340d | v341 | delta |
|---|---:|---:|---:|
| macro selected PSNR gain | 0.301510278 | 0.302505694 | +0.000995415 |
| macro selected SSIM gain | 0.003461710 | 0.003469209 | +0.000007500 |
| macro oracle headroom | 0.012505689 | 0.012075390 | -0.000430299 |
| positive PSNR-headroom views | 65 | 68 | +3 |

Main positive result:

- `room`: selected PSNR gain improves from `0.444247549` to `0.450326866`
  (`+0.006079317`), and selected SSIM gain improves by `+0.000052399`.
- On `room`, the new `source_trust` candidate is itself strong:
  `0.444884400` PSNR gain and `0.005164350` SSIM gain, beating the plain
  `learned` candidate on both axes at the scene-summary level.

Main remaining failures:

- `bonsai` regresses slightly (`-0.000106826` PSNR gain,
  `-0.000007401` SSIM gain).
- `stump` and `treehill` remain unchanged in the safe v341 run.
- The dominant remaining misses are still fixed/learned arbitration failures:
  `bonsai/00035`, `treehill/00011`, `treehill/00016`,
  `stump/00014`, `stump/00002`, `stump/00000`.

## Implemented Method Change

File:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New opt-in candidate:

```text
--enable_source_trust_residual_candidate
--source_trust_residual_candidate_name source_trust
```

The new candidate is generated inside `_candidate_deltas`, so both
source-heldout selection and target-time policies can see it. It is not a
target-only post-processing trick.

Mechanism:

- Start from the fixed source-evidence residual and the learned residual.
- Compute per-pixel trust from source confidence, support count, residual
  stability, fixed/learned alignment, and fixed/learned disagreement.
- Allow the blend to reach learned residual strength (`max_blend=1.0`) only
  where the source evidence is stable and aligned.
- Shrink toward fixed where learned residuals overshoot unstable or
  contradictory source evidence.

This is motivated by uncertainty-aware residual supervision and reliability
ideas in recent 3DGS/NBV work such as SA-ResGS
(`https://arxiv.org/html/2601.03024v1`) and uncertainty-aware 3DGS view
selection such as POp-GS
(`https://openaccess.thecvf.com/content/CVPR2025/papers/Wilson_POp-GS_Next_Best_View_in_3D-Gaussian_Splatting_with_P-Optimality_CVPR_2025_paper.pdf`),
but it is implemented here as a target-GT-free residual-candidate generator
inside the MeshSplatting/SPCarNet evaluation pipeline.

## Commands And Artifacts

Primary output root:

```text
outputs/carnet/spcarnet_v341_source_trust_focus4_20260701
```

Despite the historical directory name, it now contains the full focus6 set:

```text
bicycle
bonsai
kitchen
room
stump
treehill
```

Primary analysis artifacts:

```text
docs/car_model/results/v341_source_trust_focus6_oracle_gap.json
docs/car_model/results/v341_source_trust_focus6_oracle_gap.md
docs/car_model/results/v341_source_trust_focus4_oracle_gap.json
docs/car_model/results/v341_source_trust_focus4_oracle_gap.md
```

All v341 scene runs used offline W&B logging, for example:

```text
outputs/carnet/spcarnet_v341_source_trust_focus4_20260701/room/wandb/offline-run-20260701_081111-s1ozvbzb
```

Primary command shape:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
--base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
--base_method_name ours_26000_phasef_extra_compact_base \
--checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
--output_dir outputs/carnet/spcarnet_v341_source_trust_focus4_20260701/<scene> \
--policy_profile v322c_incumbent \
--enable_adaptive_residual_candidate \
--enable_source_trust_residual_candidate \
--no-generated_candidate_disable_when_scene_fixed \
--generated_candidate_require_source_summary_safe \
--generated_candidate_min_source_summary_psnr_delta_vs_scene -0.0005 \
--generated_candidate_min_source_summary_ssim_delta_vs_scene -0.0001 \
--enable_pairwise_dominance_policy \
--enable_source_oracle_knn_policy \
--source_oracle_knn_require_reliability_agreement \
--enable_target_neighbor_consistency_certificate \
--enable_target_neighbor_candidate_unlock \
--enable_target_neighbor_all_candidate_diagnostic \
--copy_gt --enable_wandb
```

Static checks passed:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/apply_source_heldout_support_transport_calibrator.py
git diff --check -- scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

## Negative Probes

### v341b unsuppressed generated-candidate probe

Output:

```text
outputs/carnet/spcarnet_v341b_source_trust_unsuppressed_probe_20260701
```

Purpose:

- Test whether `stump/treehill` were blocked only by source-summary suppression.

Result:

- `stump`: unchanged at fixed; `source_trust` candidate improves PSNR but does
  not beat learned/mix0750 and does not become selected.
- `treehill`: allowing generated candidates without source-summary suppression
  reduces selected PSNR from v340d `0.118121383` to `0.107348449`.

Lesson:

- The fixed-scene bottleneck is not solved by simply unsuppressing generated
  candidates. Source-summary safety is necessary.

### v342 per-view risk-model fixed-scene probe

Output:

```text
outputs/carnet/spcarnet_v342_fixed_scene_risk_unlock_probe_20260701
```

Result:

- `stump` fails badly: selected PSNR gain drops from fixed `0.057029761` to
  `0.046056597`, and SSIM also drops.
- `treehill` remains worse than v340d: selected PSNR `0.101221206` vs v340d
  `0.118121383`.

Lesson:

- The existing per-view risk model is not a safe fixed-scene unlock. It releases
  false positives when the scene-level selector falls back to fixed.

### v342b pairwise micro-relax probe

Output:

```text
outputs/carnet/spcarnet_v342b_pairwise_micro_relax_probe_20260701/stump
```

Result:

- Relaxing the global pairwise source-PSNR gate from `0.0` to `-0.0001` has no
  target effect on `stump`.

Lesson:

- The blocker is deeper than one global threshold. A reliable fixed-scene policy
  needs explicit local evidence/certificate redesign.

## Current Bottleneck

The current method is strongest when the new residual candidate is admitted and
the source reliability model can arbitrate it (`room`). The method is still weak
when the scene selector falls back to fixed and the source reliability/pairwise
policies either disable themselves or reject promotions.

The important conclusion is that `v341` validates candidate-generation as the
right direction, while the remaining work is a principled fixed-scene
micro-unlock policy, not a broader relaxation of existing gates.

Next recommended implementation:

1. Add a fixed-scene-only policy that is trained/evaluated on source-heldout
   examples but uses a stricter local certificate than the current risk model.
2. Require candidate-local PSNR support, bounded SSIM loss, and no large local
   negative tail, but do not rely on target-neighbor base-MAE alone because it
   misranks several `stump` oracle-improving views.
3. Validate first on `stump/treehill`, then rerun focus6 and compare against
   `v340d` and clean MeshSplatting baseline.

Final status: NOT COMPLETE.
