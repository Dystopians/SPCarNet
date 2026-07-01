# v331 Promotion Rollback Certificate Probe

Date: 2026-07-01

## Purpose

v329b improves full9 slightly, but treehill still has negative changed views.
The v331 probe asks whether a stronger target-blind post-decision certificate
can detect unsafe pairwise promotions before saving the target render.

This is a real pipeline change, but it is **not promoted** as the current best
method because the first focused experiments did not improve over v329b.

## Implemented Interface

Main file:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New opt-in CLI:

```text
--enable_promotion_rollback_certificate
--promotion_rollback_mode {shadow,enforce}
--promotion_rollback_sources pairwise
--promotion_rollback_min_calibration_samples
--promotion_rollback_calibration_quantile
--promotion_rollback_calibration_scale
--promotion_rollback_min_lcb_objective_delta
--promotion_rollback_min_lcb_psnr_delta
--promotion_rollback_min_lcb_ssim_delta
--promotion_rollback_min_local_cvar_delta
--promotion_rollback_min_local_min_delta
--promotion_rollback_max_local_negative_fraction
```

Default behavior is unchanged. The certificate is disabled unless
`--enable_promotion_rollback_certificate` is passed.

## Mechanism

The certificate runs after source reliability, KNN, local support, risk model,
and pairwise dominance have proposed a per-view output, but before the selected
image is saved. It does not read target/test GT.

For pairwise promotions, it uses source-heldout pairwise leave-one-out evidence
to estimate over-prediction residuals. At target time it subtracts a calibrated
source over-prediction bound from the predicted candidate-vs-incumbent margin,
then checks:

- LCB objective / PSNR / SSIM deltas versus incumbent;
- source-local CVaR and min deltas;
- source-local negative fraction.

`shadow` mode records which views would be rolled back. `enforce` mode actually
rolls the output back to the incumbent variant.

## Focused Shadow Probe

Command shape:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/treehill/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --output_dir outputs/carnet/spcarnet_v331_promotion_rollback_shadow_treehill_20260701 \
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
  --copy_gt \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v331_promotion_rollback_shadow_treehill
```

Treehill W&B offline run:

```text
outputs/carnet/spcarnet_v331_promotion_rollback_shadow_treehill_20260701/wandb/offline-run-20260701_031410-omwaz5lq
```

Stump W&B offline run:

```text
outputs/carnet/spcarnet_v331_promotion_rollback_shadow_stump_20260701/wandb/offline-run-20260701_031400-zqza792p
```

## Focused Results

Committed focused report JSONs:

```text
docs/car_model/results/v331_promotion_rollback_shadow_treehill_report.json
docs/car_model/results/v331_promotion_rollback_shadow_stump_report.json
docs/car_model/results/v331c_fineladder_treehill_report.json
```

| scene | mode | selected PSNR gain | selected SSIM gain | rollback decisions | verdict |
|---|---|---:|---:|---:|---|
| treehill | shadow | 0.104664074413 | 0.001673645443 | 0 | no effect versus v329b |
| stump | shadow | 0.057029761393 | 0.001208242029 | 0 | no effect versus v329b |

Treehill promotion rollback policy summary:

```text
enabled: true
source_sample_count: 44
keep_count: 18
shadow_rollback_count: 0
rollback_count: 0
reason_counts: {'no_promotion': 11, 'passed': 7}
calibration_error_bounds:
  objective_delta_vs_incumbent: 0.024295452939
  psnr_delta_vs_incumbent: 0.022745993536
  ssim_delta_vs_incumbent: 0.000637108948
```

Important failure: the target-blind pairwise evidence is over-confident on the
known-bad treehill views. For example, `00007` and `00009` are true negative
after target evaluation, but their source-local pairwise evidence remains
positive and clears the calibrated LCB checks.

| view | output | true PSNR delta vs fixed | true SSIM delta vs fixed | v331 decision |
|---|---|---:|---:|---|
| 00002 | mix0250 | +0.007259107 | +0.000268817 | keep |
| 00004 | mix0250 | +0.002510692 | -0.000219762 | keep |
| 00007 | mix0250 | -0.026469408 | -0.000042915 | keep |
| 00008 | mix0250 | -0.004945773 | -0.000321209 | keep |
| 00009 | mix0250 | -0.012384636 | -0.000003994 | keep |
| 00011 | mix0250 | +0.033720387 | +0.000268817 | keep |
| 00015 | mix0250 | +0.015076870 | +0.000209391 | keep |

## Fine-Ladder Probe

A second probe added `mix0125` to the fixed candidate ladder to see whether a
smaller residual step would preserve treehill positives while reducing
negatives. Because `v322c_incumbent` currently overwrites the ladder, this probe
manually reproduced the profile and used:

```text
--enable_candidate_ladder
--candidate_ladder_blends 0.125,0.25,0.75
```

W&B offline run:

```text
outputs/carnet/spcarnet_v331c_fineladder_treehill_20260701/wandb/offline-run-20260701_031709-k54oivtn
```

Result:

| method | treehill PSNR gain | treehill SSIM gain | verdict |
|---|---:|---:|---|
| v329b / v331 shadow | 0.104664074413 | 0.001673645443 | current reference |
| v331c fine ladder | 0.103565986827 | 0.001683145761 | PSNR down, SSIM up |

The fine ladder is mixed, not an improvement. It should not be promoted.

## Lessons

- Source-heldout LCB calibration alone does not catch treehill's bad promoted
  views. The source proxy distribution predicts these views as safe.
- Existing local-support relaxation already failed in v330; this v331 result
  gives a second, more specific explanation: the source-side evidence itself is
  not discriminative enough for the outdoor target tail.
- Strategy agreement between source reliability and pairwise is also too blunt:
  it would roll back the bad treehill views, but it would also roll back strong
  positives like `00011` and `00015`, reducing PSNR.
- Fine residual laddering can trade PSNR for SSIM, but does not produce an
  all-axis gain.

## Verdict

Final status: NOT COMPLETE.

v331 is useful as infrastructure and diagnosis, but not as a promoted method.
The next real improvement needs additional target-blind evidence that is not
already present in source-local residual statistics, or a stronger
representation candidate whose improvements are visually and metrically larger
before policy arbitration.
