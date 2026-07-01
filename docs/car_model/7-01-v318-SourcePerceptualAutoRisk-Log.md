# v318 Source-Perceptual Selector and Auto-Risk Margin Log

Date: 2026-07-01

## Purpose

This run tested whether the latest reflection can become an actual stronger
policy rather than another parameter scan. The design goal was a frozen,
source-only selector that can use train-heldout perceptual evidence and an
automatic risk-model margin before target/test rendering starts.

## Implemented Method Change

Code changed:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New interfaces:

```text
--compute_source_perceptual
--source_perceptual_max_side
--source_objective_lpips_weight
--source_objective_dists_weight
--per_view_risk_model_use_source_perceptual_objective
--per_view_risk_model_auto_objective_margin
--per_view_risk_model_max_accept_fraction
--per_view_risk_model_source_cvar_weight
--per_view_risk_model_source_min_weight
--per_view_risk_model_source_positive_weight
```

What changed:

- source-heldout selector can compute LPIPS and DISTS on train-heldout views;
- source objective can combine PSNR, SSIM, LPIPS gain, and DISTS gain;
- KNN policy uses the same source objective when ranking source candidates;
- risk model keeps PSNR/SSIM objective by default, with optional perceptual
  objective usage;
- risk model can search a source-heldout objective margin automatically and
  record all tested margins;
- reports now explicitly record that target/test GT is not used for selection.

Selection protocol emitted in every v318e report:

```text
scope: source_heldout_before_target_loop
target_gt_used_for_selection: false
selection_frozen_before_target_loop: true
target_gt_first_read_stage: post_render_eval_after_selected_image_save
source_perceptual_uses_train_heldout_gt_only: true
```

## Commands and Artifacts

Full9 v318e target/test apply root:

```text
outputs/carnet/spcarnet_v318e_source_perceptual_autorisk_multiscene_20260701
```

The run used W&B offline logging for every scene. It was launched in three
parallel GPU workers over:

```text
bicycle,bonsai,counter
flowers,garden,kitchen
room,stump,treehill
```

Shared command template:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<gpu> PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  --model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --output_dir outputs/carnet/spcarnet_v318e_source_perceptual_autorisk_multiscene_20260701/<scene> \
  --split test \
  --support_source_mode source_split \
  --heldout_stride 4 \
  --heldout_offset 0 \
  --device cuda \
  --k 4 \
  --alpha 0.25 \
  --learned_scale 0.5 \
  --hybrid_blend 0.5 \
  --output_variant source_heldout_auto \
  --selector_val_stride 3 \
  --selector_val_offset 0 \
  --evidence_max_side 256 \
  --compute_ssim \
  --ssim_max_side 256 \
  --compute_source_perceptual \
  --source_perceptual_max_side 256 \
  --source_objective_lpips_weight 20 \
  --source_objective_dists_weight 20 \
  --enable_per_view_knn_policy \
  --per_view_knn_k 3 \
  --per_view_knn_min_score_delta_vs_scene 0.0005 \
  --per_view_knn_forbid_fixed_when_scene_nonfixed \
  --per_view_knn_reject_variant scene \
  --per_view_knn_min_source_cvar_delta 0.0 \
  --per_view_knn_min_source_min_delta 0.0 \
  --per_view_knn_min_source_positive_fraction_delta 0.0 \
  --enable_per_view_risk_model_policy \
  --per_view_risk_model_feature_grid covered_fraction,mean_abs_delta,confidence_mean,residual_std_mean,delta_snr,signal_snr,confidence_snr,changed_fraction,delta_signal_cosine,opposition_fraction,aligned_fraction,delta_to_signal_ratio,std_to_signal_ratio,support_confidence,support_count_mean \
  --per_view_risk_model_only_when_scene_fixed \
  --per_view_risk_model_allow_when_scene_fixed \
  --per_view_risk_model_reject_variant scene \
  --per_view_risk_model_auto_objective_margin \
  --per_view_risk_model_min_source_cvar_delta 0.0 \
  --per_view_risk_model_min_source_min_delta 0.0 \
  --per_view_risk_model_min_source_positive_fraction_delta 0.0 \
  --per_view_risk_model_min_predicted_psnr_delta_vs_scene 0.0 \
  --per_view_risk_model_min_predicted_ssim_delta_vs_scene 0.0 \
  --per_view_risk_model_enable_ood_guard \
  --per_view_risk_model_ood_quantile 0.8 \
  --save_example_views 1 \
  --copy_gt \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v318e-source-perceptual-autorisk-<scene>
```

Full9 perceptual and qualitative comparison:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_support_transport_frontier_comparison.py \
  --method clean26000=outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --method v305=outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630 \
  --method v315d=outputs/carnet/spcarnet_v315d_no_fixed_downgrade_multiscene_20260701 \
  --method v316c=outputs/carnet/spcarnet_v316c_source_tail_acceptance_fixed_multiscene_20260701 \
  --method v318e=outputs/carnet/spcarnet_v318e_source_perceptual_autorisk_multiscene_20260701 \
  --output_dir outputs/carnet/spcarnet_v318e_source_perceptual_autorisk_frontier_comparison_20260701 \
  --scenes bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill \
  --panel_scenes garden,flowers,bicycle \
  --max_panels_per_scene 2 \
  --lpips_max_side 512 \
  --panel_max_side 640 \
  --crop_size 256 \
  --device cuda \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v318e-source-perceptual-autorisk-full9-comparison
```

Saved evidence:

```text
docs/car_model/results/v318e_apply_metrics_vs_prior_summary.json
docs/car_model/results/v318e_source_perceptual_autorisk_frontier_summary.json
docs/car_model/results/v318e_source_perceptual_autorisk_frontier_summary.md
docs/car_model/results/v318e_frontier_panels/
```

## Apply Metrics vs Prior Policies

| method | mean PSNR gain | mean SSIM gain | mean min PSNR | mean CVaR10 PSNR | neg views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v305 | +0.266578 | +0.003701 | +0.013917 | +0.039504 | 8 | 9/9 |
| v315d | +0.269175 | +0.003718 | +0.014301 | +0.039726 | 8 | 9/9 |
| v316c | +0.268444 | +0.003710 | +0.013917 | +0.039504 | 8 | 9/9 |
| v318e | +0.268629 | +0.003715 | +0.013917 | +0.039504 | 8 | 9/9 |

Scene-level v318e notes:

- `bicycle`: +0.119802 PSNR / +0.003016 SSIM;
- `flowers`: +0.090685 PSNR / +0.004065 SSIM;
- `treehill`: +0.099679 PSNR / +0.001643 SSIM;
- `stump`: risk policy correctly fell back to fixed, preserving safe-scene
  status.

## Clean MeshSplatting Frontier Metrics

| method | PSNR | MAE | LPIPS | DISTS | dPSNR vs clean | dMAE vs clean | dLPIPS vs clean | dDISTS vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| v305 | 27.578504 | 0.028198 | 0.087748 | 0.057662 | +0.384861 | -0.000915 | -0.002459 | -0.002240 |
| v315d | 27.582989 | 0.028182 | 0.087739 | 0.057679 | +0.389346 | -0.000930 | -0.002469 | -0.002223 |
| v316c | 27.580930 | 0.028183 | 0.087745 | 0.057673 | +0.387287 | -0.000930 | -0.002463 | -0.002229 |
| v318e | 27.581262 | 0.028185 | 0.087743 | 0.057674 | +0.387619 | -0.000928 | -0.002464 | -0.002228 |

Qualitative panels:

- [bicycle 00000](results/v318e_frontier_panels/panels/bicycle/00000_frontier_panel.png)
- [bicycle 00005](results/v318e_frontier_panels/panels/bicycle/00005_frontier_panel.png)
- [flowers 00010](results/v318e_frontier_panels/panels/flowers/00010_frontier_panel.png)
- [flowers 00014](results/v318e_frontier_panels/panels/flowers/00014_frontier_panel.png)
- [garden 00006](results/v318e_frontier_panels/panels/garden/00006_frontier_panel.png)
- [garden 00017](results/v318e_frontier_panels/panels/garden/00017_frontier_panel.png)

## Verdict

Reflection did help, but it did not create a new final method.

Positive:

- v318e is a real train/eval pipeline change, not just post-hoc reporting;
- target/test GT is explicitly excluded from selection;
- source-heldout perceptual scoring and risk auto-margin are now wired into the
  policy reports;
- the earlier treehill failure was diagnosed as a missing rich-risk feature and
  prediction-margin configuration, then fixed.

Negative:

- v318e does not beat v315d on mean PSNR, SSIM, MAE, LPIPS, or mean tail;
- v318e only slightly improves over v316c on apply PSNR/SSIM and is slightly
  worse on DISTS;
- min/CVaR/negative-view statistics are unchanged from v316c;
- visual panels still show subtle full-frame differences.

Current status:

```text
Final status: NOT COMPLETE.
```

The next useful step is not another broad weight scan. The bottleneck is that
source-heldout perceptual gains are too flat/noisy to choose a globally better
policy than v315d. A stronger next attempt should learn a reliability model that
predicts whether support-transport residuals will remain perceptually and
geometrically valid under target viewpoints, and it must be validated against
v315d as the current mean-quality frontier.
