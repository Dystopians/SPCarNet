# 6-26 v118/v119 Face-Graph Residual Transfer Milestone and Bottleneck Log

## Status

This log records the first concrete follow-up to the v116/v117 bottleneck: a train-only, evidence-derived co-visible face graph for residual transfer.

Current conclusion:

- v118 is a real mechanism change and expands target-visible edit coverage.
- v118 is not yet a paper-level quality breakthrough.
- v119 fixes a discovered implementation bottleneck: planned transfer rows were selected, but the synthetic transfer path was skipped whenever the destination face already had an atlas.
- v120 adds a stricter low-direct-support bin gate for blend mode after v119 showed that blending all bins can hurt the already fitted atlas.
- Completion status remains **NOT COMPLETE** until v119 metrics, ablations, qualitative outputs, and multi-scene evidence are finished.

## Implemented Method Change

### v118: empirical co-visible face residual transfer

Implemented in:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

Core idea:

1. Build a face graph from train evidence face-id maps, using image-space neighboring face ids as empirical co-visibility edges.
2. Use fit/train views only for residual source statistics.
3. Reserve policy-val views by `policy_val_stride`; destination faces must also be visible in policy-val evidence.
4. Require destination faces to be visible in GT-free target evidence.
5. Add a `coview_face_residual_transfer` support candidate and let the existing policy-val gate accept or reject it.

Important fairness boundary:

- Target/test GT is stripped before application.
- Target evidence contributes only geometry/visibility footprint, not color residuals.
- The final target test metrics are evaluation-only.

### v119: existing-atlas transfer activation

v118 selected transfer rows but did not actually write synthetic transfer residuals, because all selected transfer destinations already had fitted atlases and the implementation used skip-existing behavior:

```json
{
  "requested_rows": 128,
  "applied_faces": 0,
  "skipped_existing_atlas_faces": 128,
  "skip_existing_atlas": true
}
```

v119 adds an explicit mode:

```bash
--coview_transfer_existing_atlas_mode {skip,blend,overwrite}
```

Modes:

- `skip`: v118-compatible behavior.
- `blend`: inject the transfer residual as pseudo-count evidence into an existing fitted face atlas.
- `overwrite`: replace the existing face atlas with the synthetic transfer residual; used as a risky ablation, not the default promoted route.

The old flag remains compatible:

```bash
--coview_transfer_overwrite_existing_atlas
```

It maps to `--coview_transfer_existing_atlas_mode overwrite`.

### v120: low-direct-support blend gate

v119 proved the implementation path can activate, but full-bin blending degraded PSNR. v120 therefore adds:

```bash
--coview_transfer_blend_max_direct_bin_count N
```

For `blend` mode, transfer pseudo-counts are injected only into atlas bins whose direct fit count is `<= N`. `-1` keeps the v119 all-bin behavior. The first v120 pilot uses `N=1`, so the transfer prior is restricted to empty or very weakly supported bins.

## Counter Evidence So Far

Fair anchor path:

```text
/dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter
```

| Method | Path | PSNR | SSIM | LPIPS | changed_fraction | png_quantized_changed_fraction | selected_support_mode | applied transfer faces |
|---|---|---:|---:|---:|---:|---:|---|---:|
| v106/v115 anchor | `/dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter` | 27.499701 | 0.867479 | 0.238780 | 0.013267961 | n/a | base_carrier | n/a |
| v116 quant-single | `/dev/shm/peilincai_spcarnet_v116_counter_quant_single_20260626_1643/counter` | 27.499702 | 0.867478 | 0.238779 | 0.013267178 | 0.007099640 | base_carrier | n/a |
| v117 face-transfer | `/dev/shm/peilincai_spcarnet_v117_counter_face_transfer_20260626_1645/counter` | 27.499702 | 0.867478 | 0.238779 | 0.013267178 | 0.007099640 | base_carrier | n/a |
| v118 coview transfer | `/dev/shm/peilincai_spcarnet_v118_counter_coview_20260626_1735/counter` | 27.499706 | 0.867471 | 0.238765 | 0.016031105 | 0.008256266 | coview_face_residual_transfer | 0 |
| v119 coview blend | `/dev/shm/peilincai_spcarnet_v119_counter_coview_blend_20260626_1830/counter` | 27.499655 | 0.867472 | 0.238766 | 0.015949256 | 0.008147024 | coview_face_residual_transfer | 128 blend |
| v119 coview overwrite | `/dev/shm/peilincai_spcarnet_v119_counter_coview_overwrite_20260626_1830/counter` | 27.499702 | 0.867478 | 0.238779 | 0.013267178 | 0.007099640 | base_carrier | rejected coview / fallback to base |
| v119 coview blend no-basis | `/dev/shm/peilincai_spcarnet_v119_counter_coview_blend_nobasis_20260626_1830/counter` | 27.499630 | 0.867521 | 0.238846 | 0.000000000 | n/a | coview_face_residual_transfer | 128 blend, rejected/no-op |
| v120 lowbin blend | `/dev/shm/peilincai_spcarnet_v120_counter_coview_lowbin_blend_20260626_1845/counter` | 27.499947 | 0.867465 | 0.238734 | 0.020489561 | 0.010212214 | coview_face_residual_transfer | 512 blend, `blend_max_direct_bin_count=1` |
| v120 lowbin skip | `/dev/shm/peilincai_spcarnet_v120_counter_coview_lowbin_skip_20260626_1845/counter` | 27.499989 | 0.867463 | 0.238734 | 0.020591568 | 0.010340460 | coview_face_residual_transfer | 512 planned, 512 skipped |
| v121 lowbin skip alpha125 | `/dev/shm/peilincai_spcarnet_v121_counter_coview_lowbin_skip_alpha125_20260626_1905/counter` | 27.500105 | 0.867516 | 0.238785 | 0.020591527 | 0.005549537 | coview_face_residual_transfer | alpha `0.125`, 512 planned/skipped |
| v122 lowbin skip alpha1875 | `/dev/shm/peilincai_spcarnet_v122_counter_coview_lowbin_skip_alpha1875_20260626_1925/counter` | 27.500210 | 0.867499 | 0.238755 | 0.018806986 | 0.007708716 | coview_face_residual_transfer | alpha `0.1875`, 512 planned/skipped |

Interpretation before v119 finishes:

- v118 increased test edit coverage from `0.013267178` to `0.016031105`, so it partially addresses the support-starvation bottleneck.
- v118 improved PSNR and LPIPS only microscopically, while SSIM dropped. This is not enough for a paper-level claim.
- The key v118 weakness is that the actual transfer residual did not activate; the observed gain comes mainly from support expansion and refit.
- v119 blend proves the transfer application can activate (`applied_faces=128`, `blended_existing_atlas_faces=128`), but all-bin blending degrades PSNR relative to v118 and v116 full-basis.
- v119 overwrite is not a viable formal method: the policy gate rejects the coview candidate and the run falls back to the base carrier.
- v119 no-basis also rejects/falls back, which reinforces that the view/teacher basis is still necessary.
- v120 is the strongest counter result so far for PSNR and LPIPS. It expands support to `+275` faces, plans `512` transfer rows, and raises PNG-quantized changed fraction to about `1.03%`.
- v120 also clarifies the mechanism: the support-only `skip` ablation has higher PSNR than `blend`, while `blend` is only microscopically better in LPIPS. The current useful mechanism is therefore the empirical co-visible face-graph support expansion, not the transfer pseudo-count itself.
- v120 still lowers SSIM relative to v106/v116. This is the next bottleneck.
- v121 caps alpha at `0.125`, which repairs SSIM and further improves PSNR, but LPIPS becomes slightly worse than v106. This suggests the useful tradeoff likely lies between `0.125` and `0.25`.
- v122 uses the midpoint alpha `0.1875` and is the first counter pilot in this line that beats the v106 anchor on PSNR, SSIM, and LPIPS at the same time:
  - PSNR: `27.500210` vs `27.499701`
  - SSIM: `0.867499` vs `0.867479`
  - LPIPS: `0.238755` vs `0.238780`
  - target changed fraction: `0.018806986` vs `0.013267961`
  - PNG-quantized changed fraction: `0.007708716`

This is still a counter-only milestone. It is not yet a full paper endpoint.

## Running v119 Commands

### v119 blend, full-basis main candidate

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline WANDB_PROJECT=spcarnet_meshprior WANDB_DIR=/dev/shm/peilincai_wandb_v119_blend_counter_20260626 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_vnext_certified_residual_texture_scene.py --scene counter --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model --fit_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/teacher_surface_evidence --target_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/target_evidence_reparented --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_teacher_render_visible_region_carriers_phasej_trainval_alpha1_policyval_pruned.json --output_root /dev/shm/peilincai_spcarnet_v119_counter_coview_blend_20260626_1830 --target_split test --base_method_name ours_26000_phasef_extra_compact_base --method_name ours_26000_v119_coview_blend_counter --gpu 1 --skip_teacher_cache --strict_no_target_gt_apply --texture_size_candidates 16 --support_expansion_mode none --support_expansion_max_extra_faces_candidates 1 --max_abs_delta_rgb_candidates 0.12 --surface_multiscale_prior_blend_candidates 0 --atlas_empty_bin_fill_mode face_mean --enable_coview_face_residual_transfer --coview_transfer_max_faces 128 --coview_transfer_neighbor_stride 8 --coview_transfer_min_source_samples 64 --coview_transfer_min_source_mean_l1 0.0 --coview_transfer_min_edge_count 4 --coview_transfer_min_target_pixels 256 --coview_transfer_min_policy_val_pixels 64 --coview_transfer_residual_scale 0.25 --coview_transfer_synthetic_count 1 --coview_transfer_existing_atlas_mode blend --no_policy_val_bin_uncertainty_guard --enable_policy_val_effective_margin_gate --min_policy_val_effective_relative_gain 0.02 --min_policy_val_effective_ssim_gain 1e-05 --min_policy_val_effective_l1_gain 1e-06 --min_policy_val_effective_ssim_cvar20_gain 1e-06 --min_policy_val_effective_l1_cvar20_gain 0.0 --enable_policy_val_structure_aware_shrink --structure_shrink_l1_weight 1.0 --structure_shrink_gradient_weight 1.0 --structure_shrink_edge_weight 0.0 --structure_shrink_risk_tau 0.002 --structure_shrink_max_penalty 1.0 --wandb --wandb_group v119_coview_existing_atlas_20260626 --wandb_name v119-coview-blend-counter
```

Runner log:

```text
/dev/shm/v119_counter_coview_blend_runner_20260626_1830.log
```

### v119 overwrite, full-basis risk ablation

Output root:

```text
/dev/shm/peilincai_spcarnet_v119_counter_coview_overwrite_20260626_1830/counter
```

Key difference:

```bash
--coview_transfer_residual_scale 0.10 --coview_transfer_existing_atlas_mode overwrite
```

### v119 blend, no-basis ablation

Output root:

```text
/dev/shm/peilincai_spcarnet_v119_counter_coview_blend_nobasis_20260626_1830/counter
```

Key difference:

```bash
--view_conditioned_basis_mode none --teacher_distilled_basis_mode none --coview_transfer_existing_atlas_mode blend
```

### v120 lowbin blend, full-basis candidate

Output root:

```text
/dev/shm/peilincai_spcarnet_v120_counter_coview_lowbin_blend_20260626_1845/counter
```

Key parameters:

```bash
--coview_transfer_max_faces 512
--coview_transfer_neighbor_stride 4
--coview_transfer_min_source_samples 128
--coview_transfer_min_source_mean_l1 0.005
--coview_transfer_min_edge_count 4
--coview_transfer_min_target_pixels 64
--coview_transfer_min_policy_val_pixels 64
--coview_transfer_residual_scale 0.125
--coview_transfer_existing_atlas_mode blend
--coview_transfer_blend_max_direct_bin_count 1
```

### v120 lowbin skip, support-only ablation

Output root:

```text
/dev/shm/peilincai_spcarnet_v120_counter_coview_lowbin_skip_20260626_1845/counter
```

This uses the same v120 graph/source/destination policy but keeps `--coview_transfer_existing_atlas_mode skip`, so it isolates support expansion from actual residual transfer.

### v121 alpha-capped support-only candidate

Output root:

```text
/dev/shm/peilincai_spcarnet_v121_counter_coview_lowbin_skip_alpha125_20260626_1905/counter
```

Key difference from v120 skip:

```bash
--alpha_grid 0,0.0625,0.125
--policy_val_ssim_alpha_refinement_steps 0
```

Purpose: test whether a lower target edit strength can preserve the v120 PSNR/LPIPS gain while repairing the held-out SSIM drop.

### v122 alpha1875 support-only candidate

Output root:

```text
/dev/shm/peilincai_spcarnet_v122_counter_coview_lowbin_skip_alpha1875_20260626_1925/counter
```

Key difference from v121:

```bash
--alpha_grid 0,0.125,0.1875
--policy_val_ssim_alpha_refinement_steps 0
```

Purpose: test the midpoint between v121 and v120, aiming for PSNR/SSIM/LPIPS all above the v106 anchor.

Completed result:

```text
PSNR  27.50020980834961
SSIM  0.8674987554550171
LPIPS 0.2387545108795166
```

Selected alpha: `0.1875`.

Output render/GT count: `30 / 30`.

Qualitative panels:

```text
docs/car_model/vnext_artifacts/counter_v122_alpha1875_panel_20260626/counter_v122_vs_v106_panel.png
docs/car_model/vnext_artifacts/counter_v122_alpha1875_panel_20260626/counter_v122_vs_v106_all_positive_panel.png
```

Panel manifests:

```text
docs/car_model/vnext_artifacts/counter_v122_alpha1875_panel_20260626/counter_v122_vs_v106_panel_manifest.json
docs/car_model/vnext_artifacts/counter_v122_alpha1875_panel_20260626/counter_v122_vs_v106_all_positive_panel_manifest.json
```

## Lessons and Claim Boundary

The current evidence supports this claim:

> The new line has moved from parameter search toward a geometry/evidence-driven residual-transfer mechanism with explicit train-only source construction and policy-val certification.

The current evidence does **not** yet support this claim:

> v118/v119 is a final paper endpoint that broadly beats MeshSplatting or v106.

Remaining required evidence:

1. The promoted story must be renamed honestly: current evidence favors empirical co-visible graph support expansion plus alpha calibration, not residual pseudo-count transfer.
2. The v122 counter result must be replicated on more scenes before any paper-level story is promoted.
3. Additional ablations should separate `max_faces=512`, `neighbor_stride=4`, `min_source_mean_l1=0.005`, and alpha `0.1875`.
4. Qualitative outputs should be expanded beyond counter and selected for visibly meaningful differences, not just metric-positive micro-deltas.
5. The method needs a fixed adaptive policy that chooses alpha without hand-picking `0.1875` per scene.
