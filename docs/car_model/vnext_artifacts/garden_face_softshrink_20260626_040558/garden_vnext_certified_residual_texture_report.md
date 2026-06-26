# vNext Certified Residual Surface Texture Run

- method: `vNext_certified_residual_surface_texture`
- scene: `garden`
- status: `COMPLETE`
- run root: `/dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558/garden`
- protocol audit passed: `True`
- target split: `test`
- selection uses test GT: `False`
- capacity selected on: `train_policy_val_and_gt_free_target_footprint`
- thresholds selected on: `train_policy_val`

## Inputs

- fit_evidence_dir: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1` exists=`True`
- parent_render_dir: `None`
- region_carrier_json: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1_policyval_pruned.json` exists=`True`
- source_model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model` exists=`True`
- target_evidence_dir: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/garden` exists=`True`
- teacher_render_dir: `None`
- texture_fit_evidence_dir: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1` exists=`True`

## Settings

- alpha_grid: `0,0.125,0.25,0.5`
- atlas_empty_bin_fill_mode: `face_mean`
- base_method_name: `ours_26000_phasef_extra_compact_base`
- bin_uncertainty_guard_min_bin_samples: `16`
- bin_uncertainty_guard_min_positive_view_fraction: `0.5`
- bin_uncertainty_shrink_fallback_shrink: `1.0`
- bin_uncertainty_shrink_min_bin_samples: `16`
- bin_uncertainty_shrink_min_positive_view_fraction: `0.5`
- bin_uncertainty_shrink_min_relative_gain: `0.0`
- bin_uncertainty_shrink_policy_mode: `keep_with_downweight`
- dry_run: `False`
- face_gain_guard_min_positive_view_fraction: `0.5`
- gpu: `2`
- max_abs_delta_rgb: `0.12`
- max_abs_delta_rgb_candidates: `0.12`
- method_name: `ours_26000_vnext_certified_residual_surface_texture`
- min_alpha: `0.03`
- min_l1: `0.0`
- min_policy_val_cvar20_relative_gain: `0.0`
- min_policy_val_l1_mean_gain: `0.0`
- min_policy_val_l1_min_view_gain: `-1e-06`
- min_policy_val_l1_positive_view_fraction: `0.55`
- min_policy_val_min_view_relative_gain: `-1e-06`
- min_policy_val_positive_view_fraction: `0.55`
- min_policy_val_relative_gain: `0.0`
- min_policy_val_samples: `1024`
- min_policy_val_ssim_mean_gain: `-1e-07`
- min_policy_val_ssim_min_view_gain: `-1e-05`
- min_policy_val_ssim_positive_view_fraction: `0.55`
- no_mask_teacher_target: `False`
- no_policy_val_bin_uncertainty_guard: `True`
- no_policy_val_bin_uncertainty_shrink: `False`
- no_policy_val_ssim_alpha_refinement: `False`
- no_preacceptance_policy_val_guard_repair: `False`
- output_root: `/dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558`
- policy_val_ssim_alpha_refinement_min_alpha: `0.001`
- policy_val_ssim_alpha_refinement_steps: `7`
- policy_val_stride: `4`
- scene: `garden`
- skip_eval: `False`
- skip_teacher_cache: `True`
- skip_texture: `False`
- support_expansion_max_extra_faces_candidates: `2048,4096`
- support_expansion_min_face_samples: `64`
- support_expansion_mode: `none`
- surface_multiscale_prior_blend_candidates: `0.5`
- surface_multiscale_prior_min_cosine: `0.0`
- surface_multiscale_prior_min_direct_samples: `1`
- surface_multiscale_prior_min_sign_consistency: `0.5`
- surface_multiscale_prior_mode: `local_patch`
- target_footprint_tail_risk_min_cvar20_view_gain: `0.0`
- target_footprint_tail_risk_min_min_view_gain: `-1e-08`
- target_footprint_tail_risk_min_positive_view_fraction: `0.75`
- target_split: `test`
- teacher_distilled_basis_blend: `0.5`
- teacher_distilled_basis_mode: `face_uv_patch_mixture_ridge`
- teacher_parent_delta_min: `0.005`
- teacher_render_error_margin: `0.0`
- teacher_selection_mode: `better_masked_residual`
- texture_size: `16`
- texture_size_candidates: `16`
- top_support_limit: `8192`
- top_support_min_alpha: `0.03`
- view_conditioned_basis_mode: `normal_camera_linear`
- wandb: `True`
- wandb_group: `vnext_face_soft_bin_shrink_pilot`
- wandb_mode: `offline`
- wandb_name: `vnext-face-softshrink-garden-20260626_040558`
- wandb_project: `spcarnet_meshprior`

## Commands

### apply_certified_residual_texture

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model --fit_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1 --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/garden --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1_policyval_pruned.json --output_model /dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558/garden/model --target_split test --base_method_name ours_26000_phasef_extra_compact_base --method_name ours_26000_vnext_certified_residual_surface_texture --residual_rgb_key teacher_residual_rgb --residual_l1_key teacher_residual_l1 --texture_size 16 --texture_size_candidates 16 --support_expansion_mode none --support_expansion_max_extra_faces_candidates 2048,4096 --support_expansion_min_face_samples 64 --policy_val_stride 4 --alpha_grid 0,0.125,0.25,0.5 --min_l1 0.0 --min_alpha 0.03 --max_abs_delta_rgb 0.12 --max_abs_delta_rgb_candidates 0.12 --atlas_empty_bin_fill_mode face_mean --surface_multiscale_prior_mode local_patch --surface_multiscale_prior_blend_candidates 0.5 --surface_multiscale_prior_gate_mode evidence_consistent --surface_multiscale_prior_min_direct_samples 1 --surface_multiscale_prior_min_sign_consistency 0.5 --surface_multiscale_prior_min_cosine 0.0 --view_conditioned_basis_mode normal_camera_linear --view_conditioned_basis_guard_mode policy_val_nonregressive --view_conditioned_basis_ood_mode diag_z --teacher_distilled_basis_mode face_uv_patch_mixture_ridge --teacher_distilled_basis_guard_mode policy_val_nonregressive --teacher_distilled_basis_apply_mode blend --teacher_distilled_basis_blend 0.5 --select_alpha_by_risk_gate --enable_policy_val_ssim_alpha_refinement --policy_val_ssim_alpha_refinement_steps 7 --policy_val_ssim_alpha_refinement_min_alpha 0.001 --enable_preacceptance_policy_val_guard_repair --min_policy_val_samples 1024 --min_policy_val_relative_gain 0.0 --min_policy_val_positive_view_fraction 0.55 --min_policy_val_cvar20_relative_gain 0.0 --min_policy_val_min_view_relative_gain=-1e-06 --enable_policy_val_image_ssim_gate --min_policy_val_ssim_mean_gain=-1e-07 --min_policy_val_ssim_positive_view_fraction 0.55 --min_policy_val_ssim_min_view_gain=-1e-05 --enable_policy_val_image_l1_gate --min_policy_val_l1_mean_gain 0.0 --min_policy_val_l1_positive_view_fraction 0.55 --min_policy_val_l1_min_view_gain=-1e-06 --enable_policy_val_face_gain_guard --face_gain_guard_min_positive_view_fraction 0.5 --enable_policy_val_bin_uncertainty_shrink --bin_uncertainty_shrink_policy_mode keep_with_downweight --bin_uncertainty_shrink_min_bin_samples 16 --bin_uncertainty_shrink_min_relative_gain=0.0 --bin_uncertainty_shrink_min_positive_view_fraction 0.5 --bin_uncertainty_shrink_fallback_shrink 1.0 --enable_target_support_candidate_selection --enable_policy_candidate_dominance_pruning --enable_policy_val_prior_bin_gain_hybrid --enable_prior_bin_gain_hybrid_l1_proxy_gate --enable_policy_val_source_mixture --enable_target_footprint_bin_certificate --enable_target_footprint_tail_risk_certificate --target_footprint_tail_risk_min_positive_view_fraction 0.75 --target_footprint_tail_risk_min_min_view_gain=-1e-08 --target_footprint_tail_risk_min_cvar20_view_gain 0.0 --write_noop_on_reject --noop_fallback_source target_evidence --force
```

- returncode: `0`
- elapsed_sec: `344.02937865257263`
- log: `/dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558/garden/logs/02_certified_texture.log`

### evaluate_vnext_target

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/evaluate_render_split_metrics.py --model_path /dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558/garden/model --split test --methods ours_26000_vnext_certified_residual_surface_texture --output /dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558/garden/reports/garden_ours_26000_vnext_certified_residual_surface_texture_test_results.json --per_view_output /dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558/garden/reports/garden_ours_26000_vnext_certified_residual_surface_texture_test_per_view.json --merge_model_results
```

- returncode: `0`
- elapsed_sec: `43.936497926712036`
- log: `/dev/shm/peilincai_spcarnet_vnext_face_softshrink_garden_20260626_040558/garden/logs/03_eval.log`

## Errors

- none
