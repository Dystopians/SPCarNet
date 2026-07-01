# v322C Candidate Ladder With Incumbent-Preserving Policy

Date: 2026-07-01

## Status

Final status: NOT COMPLETE.

v322C is a real train/eval pipeline method change and a verified full9
milestone, but not paper closure. It adds dynamic residual ladder candidates
and preserves v321G safety while giving a small positive full9/frontier gain.
The improvement is too small to claim a decisive top-conference breakthrough.

## Method Change

`scripts/car_model/apply_source_heldout_support_transport_calibrator.py` now
supports dynamic candidate residual blends:

```text
--enable_candidate_ladder
--candidate_ladder_blends 0.25,0.75
--per_view_knn_base_variants_only
```

The candidate set becomes:

```text
fixed, learned, hybrid, mix0250, mix0750
```

The implementation writes `candidate_summaries`, per-view `candidate_metrics`,
`candidate_variants`, and the candidate blend map into the apply report. Direct
`--output_variant mix0250` / `mix0750` ablations are also supported through
dynamic run-time validation.

The key v322C policy decision is incumbent preservation:

- KNN remains base-only (`fixed/learned/hybrid`) because v322B showed that KNN
  over ladder candidates mis-selected `bicycle` and `flowers` views.
- Source reliability can still use ladder candidates, but v322C uses a fixed
  low objective margin instead of auto-raising the margin. This restored the
  `bonsai` hybrid view that v322B rejected.
- The policy remains target/test GT free. Target GT is read only after selected
  renders are saved for evaluation.

## Negative Intermediate: v322B

v322B proved the ladder is meaningful but unsafe if every selector can freely
use it.

Focused 7-scene aggregate:

| method | PSNR gain | SSIM gain | mean min PSNR | mean CVaR10 | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v321G | +0.217322 | +0.003246 | +0.008899 | +0.028290 | 7 | 7/7 |
| v322B | +0.217248 | +0.003247 | +0.008649 | +0.028147 | 7 | 7/7 |

Failure causes:

- `bicycle/00002`: v322B changed `learned -> mix0750` and lost `0.027523` PSNR.
- `bicycle/00017`: v322B changed `hybrid -> mix0250` and lost `0.011251` PSNR.
- `bonsai/00005`: auto-margin changed `hybrid -> learned` and lost `0.011061` PSNR.

## v322C Full9 Apply Result

Evidence root:

```text
outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701
docs/car_model/results/v322c_baseknn_ladder_full9_vs_v321g_summary.json
```

| method | PSNR gain | SSIM gain | mean min PSNR | mean CVaR10 | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v319c | +0.269725 | +0.003720 | +0.014301 | +0.039726 | 8 | 9/9 |
| v321G | +0.271248 | +0.003727 | +0.014301 | +0.039726 | 8 | 9/9 |
| v322C | +0.271334 | +0.003727 | +0.014301 | +0.039726 | 8 | 9/9 |

v322C minus v321G:

| metric | delta |
|---|---:|
| mean PSNR gain | +0.000086 |
| mean SSIM gain | +0.000000594 |
| mean min PSNR gain | +0.000000 |
| mean CVaR10 PSNR gain | +0.000000 |
| negative views | +0 |
| safe scenes | +0 |

Scene deltas vs v321G:

| scene | PSNR delta | SSIM delta | note |
|---|---:|---:|---|
| garden | +0.000403 | +0.00000593 | ladder selected on 5 views |
| room | +0.000372 | -0.00000059 | ladder selected on 4 views |
| bicycle | +0.000000 | +0.00000000 | restored v321G behavior |
| bonsai | +0.000000 | +0.00000000 | restored v321G behavior |
| counter | +0.000000 | +0.00000000 | no regression |
| flowers | +0.000000 | +0.00000000 | restored v321G behavior |
| kitchen | +0.000000 | +0.00000000 | no regression |
| stump | +0.000000 | +0.00000000 | no regression |
| treehill | +0.000000 | +0.00000000 | no regression |

Selected variant counts:

| method | fixed | learned | hybrid | mix0250 | mix0750 |
|---|---:|---:|---:|---:|---:|
| v321G | 28 | 153 | 65 | 0 | 0 |
| v322C | 28 | 151 | 58 | 4 | 5 |

## Frontier / Qualitative Result

Evidence root:

```text
outputs/carnet/spcarnet_v322c_frontier_comparison_full9_20260701
docs/car_model/results/v322c_frontier_lpips_qualitative_summary.json
docs/car_model/results/v322c_frontier_lpips_qualitative_summary.md
docs/car_model/results/v322c_frontier_panels/
```

| method | scenes | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|
| clean26000 | 9 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v319c | 9 | 27.583642 | 0.028181 | 0.087746 | 0.057678 |
| v321G | 9 | 27.586900 | 0.028173 | 0.087736 | 0.057660 |
| v322C | 9 | 27.587073 | 0.028173 | 0.087735 | 0.057659 |

v322C minus v321G:

| metric | delta |
|---|---:|
| PSNR | +0.000173 |
| MAE | -0.000000797 |
| LPIPS | -0.000000673 |
| DISTS | -0.000001292 |

Qualitative panels were generated for:

```text
docs/car_model/results/v322c_frontier_panels/
```

## Commands

Representative v322C apply command:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --output_dir outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701/<scene> \
  --output_variant source_heldout_auto \
  --enable_candidate_ladder --candidate_ladder_blends 0.25,0.75 \
  --enable_per_view_knn_policy --per_view_knn_base_variants_only \
  --source_reliability_min_predicted_objective_delta_vs_scene -0.000000001 \
  --enable_source_reliability_policy --source_reliability_reject_variant scene \
  --source_reliability_enable_calibrated_lcb --source_reliability_calibrated_lcb_mode raw_incumbent \
  --compute_ssim --enable_wandb
```

Representative frontier command:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/build_support_transport_frontier_comparison.py \
  --method clean26000=outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --method v319c=outputs/carnet/spcarnet_v319c_incumbent_reliability_full9_20260701 \
  --method v321g=outputs/carnet/spcarnet_v321g_rawmargin_accept10_full9_20260701 \
  --method v322c=outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701 \
  --output_dir outputs/carnet/spcarnet_v322c_frontier_comparison_full9_20260701 \
  --scenes bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill \
  --clean_iteration 26000 --device cuda --enable_wandb
```

## Lessons

- The ladder candidate space is real: v322C selected `mix0250/mix0750` on 9
  target views and improved `room/garden`.
- The first naive ladder integration failed. Dynamic candidates must be
  incumbent-preserving; otherwise KNN and auto-margin can trade away robust base
  decisions for noisy source-heldout predictions.
- v322C is better than v321G on full9 mean and frontier metrics, but only by a
  very small amount. This is an engineering/reliability milestone, not a final
  paper-level breakthrough.
- The next useful step is not another broad threshold scan. It should target a
  stronger representation or a calibrated per-view model that can safely unlock
  more of the visible learned/mix oracle gap on `stump`, `treehill`, `room`, and
  `bicycle`.
