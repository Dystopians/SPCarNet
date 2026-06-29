# SPCarNet v171 Configurable-Rank Low-Rank Texture Log

Date: 2026-06-28

## Verdict

`v171` is a real representation-side implementation milestone, but it is **not** a paper-level success. It fails the v169 improved-prompt hard flowers exact gate against Phase-J, so no full9 run should be launched from this version.

Hard Phase-J flowers gate:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-J flowers reference | 20.304358 | 0.557770 | 0.329222 |
| v170 rich tail-cap exact | 19.832060 | 0.505779 | 0.405906 |
| v171 configurable-rank exact | 19.832148 | 0.505778 | 0.405912 |
| v171 - Phase-J | -0.472210 | -0.051992 | +0.076690 |

Result: **FAIL_A**. The method does not beat Phase-J on any all-axis interpretation. LPIPS is worse, SSIM is far worse, and PSNR is still below Phase-J.

## What Changed

The v169/v171 prompt required a real representation change instead of another alpha/support scan. I implemented configurable low-rank Phase-J teacher residual textures:

- `low_rank_view_texture` and `low_rank_view_texture_rich` are now generic low-rank modes, while the old `_k4` names remain compatible.
- `--teacher_distilled_low_rank_texture_rank` selects requested rank.
- `--teacher_distilled_low_rank_texture_rank_candidates` enables rank ladders in the policy candidate loop.
- The diagnostic script can sweep ranks before exact runs.
- Audit/report output now records selected rank, rank candidates, effective rank cap, mean rank, and retained energy.

Touched files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- `scripts/car_model/analyze_v169_policy_val_upper_bound.py`

Validation:

```bash
python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  scripts/car_model/analyze_v169_policy_val_upper_bound.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  scripts/car_model/analyze_v169_policy_val_upper_bound.py
```

Both checks passed.

## Storage And Runtime Preflight

The run was constrained by storage:

```text
/data    28T used almost fully, only about 6.1M free
/dev/shm 252G with about 1.1-1.2G free during the run
/tmp     root filesystem with about 6.1T free
quota    /dev/nvme0n1p4 reports 102050M used vs 100G limit
```

Therefore v171 reused pre-existing low-copy evidence under `/dev/shm` and did not materialize new full caches under `/data`.

## Policy-Val Diagnostic

Diagnostic artifacts:

- JSON: `/dev/shm/peilincai_spcarnet_v171_diagnostics/flowers_policy_val_rich_lowrank_rank_sweep_4_6_8_lpips.json`
- Markdown: `/dev/shm/peilincai_spcarnet_v171_diagnostics/flowers_policy_val_rich_lowrank_rank_sweep_4_6_8_lpips.md`

Command:

```bash
mkdir -p /dev/shm/peilincai_spcarnet_v171_diagnostics
CUDA_VISIBLE_DEVICES=2 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/analyze_v169_policy_val_upper_bound.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --region_carrier_json /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json \
  --output_json /dev/shm/peilincai_spcarnet_v171_diagnostics/flowers_policy_val_rich_lowrank_rank_sweep_4_6_8_lpips.json \
  --output_md /dev/shm/peilincai_spcarnet_v171_diagnostics/flowers_policy_val_rich_lowrank_rank_sweep_4_6_8_lpips.md \
  --texture_sizes 16 \
  --alpha_grid 0,0.0625 \
  --policy_val_stride 4 \
  --max_samples_per_view 240000 \
  --teacher_distilled_basis_mode low_rank_view_texture_rich \
  --teacher_distilled_low_rank_texture_ranks 4,6,8 \
  --teacher_distilled_basis_min_face_samples 128 \
  --teacher_distilled_basis_ridge 0.02 \
  --teacher_distilled_basis_apply_mode blend \
  --teacher_distilled_basis_blend 0.5 \
  --enable_adaptive_low_support_teacher_basis \
  --adaptive_teacher_basis_min_face_samples_floor 128 \
  --adaptive_teacher_basis_support_quantile 0.25 \
  --adaptive_teacher_basis_low_support_ridge_scale 0.5 \
  --policy_val_lpips_max_size 256
```

The diagnostic technically passed the all-axis policy-val upper-bound gate, but only by microscopic margins:

| rank | retained energy | relative gain | proxy PSNR gain | SSIM gain | LPIPS gain | LPIPS min-view gain |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.882399 | 0.038764 | +0.171698 | +0.000001108 | +0.000002189 | -0.000029892 |
| 6 | 0.947594 | 0.039215 | +0.173738 | +0.000001008 | +0.000001382 | -0.000027880 |
| 8 | 0.978499 | 0.039636 | +0.175643 | +0.000001063 | +0.000002436 | -0.000026442 |

Interpretation: the carrier can reduce a residual-sample proxy, but it does not carry enough image-level structure to move exact SSIM/LPIPS. The policy-val pass was too weak to be a strong success signal.

## Flowers Exact Run

Run root:

```text
/dev/shm/peilincai_spcarnet_20260629_v171_rich_rank_sweep_exact/flowers
```

W&B:

```text
mode: offline
dir: /dev/shm/peilincai_wandb_v171_rich_rank_sweep_exact/wandb/offline-run-20260628_212238-j86josl2
```

Main command:

```bash
WANDB_MODE=offline \
WANDB_DIR=/dev/shm/peilincai_wandb_v171_rich_rank_sweep_exact \
CUDA_VISIBLE_DEVICES=2 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene flowers \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --eval_gt_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --prestripped_target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v169_true_lowrank_k4_reuse_exact/flowers/target_evidence_no_gt \
  --region_carrier_json /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json \
  --output_root /dev/shm/peilincai_spcarnet_20260629_v171_rich_rank_sweep_exact \
  --method_name ours_26000_v171_rich_rank_sweep_flowers \
  --skip_teacher_cache \
  --strict_no_target_gt_apply \
  --wandb \
  --wandb_mode offline \
  --wandb_group v171_rich_rank_sweep_flowers \
  --wandb_name v171-rich-rank-sweep-flowers \
  --texture_size 16 \
  --texture_size_candidates 16 \
  --support_expansion_mode none \
  --support_expansion_max_extra_faces_candidates 0 \
  --target_footprint_residual_debt_match_level bin \
  --policy_val_stride 4 \
  --alpha_grid 0,0.0625 \
  --no_policy_val_ssim_alpha_refinement \
  --no_preacceptance_policy_val_guard_repair \
  --min_l1 0.0 \
  --min_alpha 0.03 \
  --max_abs_delta_rgb 0.12 \
  --max_abs_delta_rgb_candidates 0.12 \
  --atlas_empty_bin_fill_mode face_mean \
  --surface_multiscale_prior_mode none \
  --surface_multiscale_prior_blend_candidates 0 \
  --view_conditioned_basis_mode none \
  --view_cluster_expert_count 1 \
  --view_cluster_feature_mode none \
  --teacher_distilled_basis_mode low_rank_view_texture_rich \
  --teacher_distilled_low_rank_texture_rank 4 \
  --teacher_distilled_low_rank_texture_rank_candidates 4,6,8 \
  --teacher_distilled_basis_ridge 0.02 \
  --teacher_distilled_basis_blend 0.5 \
  --teacher_distilled_basis_min_face_samples 128 \
  --enable_adaptive_low_support_teacher_basis \
  --adaptive_teacher_basis_min_face_samples_floor 128 \
  --adaptive_teacher_basis_support_quantile 0.25 \
  --adaptive_teacher_basis_low_support_ridge_scale 0.5 \
  --enable_policy_val_image_lpips_gate \
  --policy_val_lpips_max_size 256 \
  --enable_policy_val_effective_margin_gate
```

Note: the actual launched command also repeated `--min_policy_val_effective_lpips_gain 1e-6`; this was harmless but should be cleaned in future command templates.

Exact artifacts:

- Manifest: `/dev/shm/peilincai_spcarnet_20260629_v171_rich_rank_sweep_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- Report: `/dev/shm/peilincai_spcarnet_20260629_v171_rich_rank_sweep_exact/flowers/reports/flowers_vnext_certified_residual_texture_report.md`
- Results: `/dev/shm/peilincai_spcarnet_20260629_v171_rich_rank_sweep_exact/flowers/reports/flowers_ours_26000_v171_rich_rank_sweep_flowers_test_results.json`
- Per-view: `/dev/shm/peilincai_spcarnet_20260629_v171_rich_rank_sweep_exact/flowers/reports/flowers_ours_26000_v171_rich_rank_sweep_flowers_test_per_view.json`
- Eval GT audit: `/dev/shm/peilincai_spcarnet_20260629_v171_rich_rank_sweep_exact/flowers/reports/flowers_ours_26000_v171_rich_rank_sweep_flowers_test_eval_gt_population_audit.json`

Protocol audit:

```text
status: COMPLETE
errors: []
protocol_audit.passed: true
target_gt_visible_to_apply: false
target_gt_visible_to_selection: false
target_gt_visible_to_eval: true
selection_uses_test_gt: false
target_forbidden_keys_stripped: true
```

Apply impact:

```text
written_views: 22
changed_pixels: 40517 / 37100800 = 0.001092
png_quantized_changed_pixels: 18723 / 37100800 = 0.000505
selected_alpha: 0.0625
selected_rank in final audit: 4
rank candidates: [4, 6, 8]
```

The final audit selected rank 4 even though rank 8 was also accepted in the candidate log. This is consistent with the current policy sorting: rank 4 has slightly higher SSIM gain than rank 8 in the selected all-axis row. The more important fact is that all rank choices have only microscopic SSIM/LPIPS gains.

## Why v171 Failed

1. **Policy-val signal is too small.** SSIM and LPIPS gains are about `1e-6` to `2e-6`, which is not a robust image-quality improvement.
2. **Target/test impact is tiny.** Only `0.0505%` of PNG pixels changed after quantization, so exact PSNR/SSIM/LPIPS are effectively unchanged from v170.
3. **The carrier is still too sparse and local.** The accepted surface support is `342` carrier faces, with selected low-rank support over `255` faces and `10633 / 65280` supported bins. This is not enough to reproduce Phase-J's view-dependent image correction.
4. **Rank expansion mainly increases retained residual energy, not perceptual structure.** Rank 8 raises retained energy to about `0.9785`, but exact metrics remain flat.
5. **This is not a full9 candidate.** The prompt explicitly forbids full9 promotion before flowers exact beats Phase-J all-axis. v171 does not satisfy that requirement.

## Next Required Direction

The next attempt should be `v172`, but it should **not** be another low-rank rank/alpha/support scan. The evidence says the current atlas carrier is underpowered. The next real method should move to a train-fit-only, target/test no-GT-safe **surface feature texture plus compact decoder/MoE**:

- train only on Phase-J teacher residual from train-fit views;
- certify with policy-val GT;
- apply to target/test using only parent render, geometry, normal, barycentric/UV, depth, alpha, and learned surface features;
- preserve strict no-target-GT apply;
- optimize an image/perceptual/structure-aware objective, not only residual MSE;
- expose a flowers exact gate before any full9.

The minimum useful v172 success condition remains unchanged:

```text
flowers exact must beat Phase-J:
PSNR > 20.304358
SSIM > 0.557770
LPIPS < 0.329222
```

Until that happens, the project status is **NOT COMPLETE** under the v169 improved prompt.

## v172 Existing MoE Check

After this v171 log was first written, I ran a narrow no-new-code check suggested by a subagent: combine the existing low-rank teacher residual texture with `view_cluster_expert_count=3` and `view_cluster_feature_mode=camera_center`. This was intended to test whether the existing train-fit-only view-cluster expert path could provide a stronger representation without writing a new decoder.

Run root:

```text
/dev/shm/peilincai_spcarnet_20260629_v172_lowrank_moe_exact/flowers
```

W&B:

```text
mode: offline
dir: /dev/shm/peilincai_wandb_v172_lowrank_moe_exact/wandb/offline-run-20260628_213455-onedghnj
```

Artifacts:

- Manifest: `/dev/shm/peilincai_spcarnet_20260629_v172_lowrank_moe_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- Results: `/dev/shm/peilincai_spcarnet_20260629_v172_lowrank_moe_exact/flowers/reports/flowers_ours_26000_v172_lowrank_moe_flowers_test_results.json`
- Per-view: `/dev/shm/peilincai_spcarnet_20260629_v172_lowrank_moe_exact/flowers/reports/flowers_ours_26000_v172_lowrank_moe_flowers_test_per_view.json`

The run completed with strict protocol audit passing:

```text
status: COMPLETE
errors: []
protocol_audit.passed: true
target_gt_visible_to_apply: false
target_gt_visible_to_selection: false
selection_uses_test_gt: false
target_forbidden_keys_stripped: true
```

However, the MoE candidate was rejected by policy-val:

```text
accepted: false
effective_policy: fallback_noop
selected_alpha: 0.0
changed_pixels: 0 / 37100800
```

Policy-val best row before rejection:

```text
relative_gain: +0.039999002
SSIM gain: +0.000001371
image L1 gain: +0.000000470
LPIPS gain: +0.000002391
LPIPS positive-view fraction: 0.416667
LPIPS min-view gain: -0.000031874
```

The decisive failure was perceptual/tail safety: LPIPS positive-view fraction was below the gate and min-view LPIPS was slightly outside the allowed tail bound.

Exact metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-J flowers reference | 20.304358 | 0.557770 | 0.329222 |
| v172 low-rank MoE no-op | 19.832010 | 0.505779 | 0.405904 |
| v172 - Phase-J | -0.472348 | -0.051991 | +0.076682 |

Interpretation:

`v172` proves that the existing view-cluster MoE switch is not enough. It cleanly rejects the candidate and falls back to no-op, so this is not a full9 candidate. The next useful step must be a new representation implementation, not a recombination of existing low-rank/MoE/gate switches.
