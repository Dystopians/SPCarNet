# vNext Counter Qualitative Panel Run Log

Date: 2026-06-26

This log records a focused counter rerun of the current fixed vNext structure-aware shrink policy with render outputs preserved. The purpose is to close a qualitative-evidence gap in the vNext package, not to promote vNext as a quality-superior endpoint.

## Outcome

Run root:

```text
/dev/shm/peilincai_spcarnet_vnext_counter_qual_20260626_125352
```

W&B offline root:

```text
/dev/shm/peilincai_wandb_vnext_counter_qual_20260626_125352/wandb/offline-run-20260626_130421-mgp6psen
```

Key result:

| item | value |
|---|---:|
| scene | `counter` |
| status | `COMPLETE` |
| protocol audit passed | `true` |
| errors | `0` |
| accepted | `true` |
| selected alpha | `0.125` |
| changed fraction | `0.012343567457579047` |
| written test views | `30` |
| apply elapsed | `467.368093 sec` |
| eval elapsed | `55.807792 sec` |

The run preserved the final vNext render and GT directories:

```text
/dev/shm/peilincai_spcarnet_vnext_counter_qual_20260626_125352/counter/model/test/ours_26000_vnext_structure_aware_shrink/renders
/dev/shm/peilincai_spcarnet_vnext_counter_qual_20260626_125352/counter/model/test/ours_26000_vnext_structure_aware_shrink/gt
```

## Fair Metrics

Fairness note: for this local counter reproduction, the clean-best MeshSplatting checkpoint is `ours_26000`, not `ours_30000`. The qualitative panel therefore uses `cleanbest26000`.

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting best (`ours_26000`) | `26.751773834` | `0.862055242` | `0.252003312` |
| compact base (`ours_26000_phasef_extra_compact_base`) | `26.749872208` | `0.862051308` | `0.251997739` |
| vNext structure-aware shrink | `26.751171112` | `0.862042248` | `0.251955062` |
| vNext - compact base | `+0.001298904` | `-0.000009060` | `-0.000042677` |
| vNext - clean best | `-0.000602722` | `-0.000012994` | `-0.000048250` |

Interpretation:

- vNext improves compact base in PSNR and LPIPS but slightly regresses SSIM.
- vNext is not a strict RGB win over clean-best MeshSplatting on this scene; it is only slightly better on LPIPS and slightly lower on PSNR/SSIM.
- This agrees with the full9 conclusion: vNext is a protocol/evidence milestone and bottleneck diagnosis, not a promoted quality endpoint.

## Qualitative Panel

Panel:

```text
docs/car_model/vnext_artifacts/counter_qualitative_panel_20260626_125352/counter_cleanbest_base_vnext_panel.png
```

Manifest:

```text
docs/car_model/vnext_artifacts/counter_qualitative_panel_20260626_125352/counter_cleanbest_base_vnext_panel_manifest.json
```

Selected frames:

```text
00000.png, 00029.png, 00027.png, 00028.png, 00009.png, 00013.png
```

The panel columns are:

```text
GT | cleanbest26000 | compact_base | vnext | |vnext-GT| x4 | |compact_base-GT| x4 | |vnext-compact_base| x4
```

The panel is technically valid and useful for slides because it shows GT, clean-best baseline, compact base, vNext, and amplified error maps under one view selection rule. It also makes the current limitation visible: `|vnext-compact_base| x4` is almost black, so the current representation edit is visually very subtle.

## Commands

Smoke test for the reusable panel exporter:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/build_vnext_qualitative_panels.py scripts/car_model/smoke_test_vnext_qualitative_panels.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_vnext_qualitative_panels.py
```

Focused counter vNext rerun:

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_DIR=/dev/shm/peilincai_wandb_vnext_counter_qual_20260626_125352 \
WANDB_MODE=offline \
PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene counter \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model \
  --fit_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/counter_teacher_surface_evidence_phasej_trainval_alpha1 \
  --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/counter \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_teacher_render_visible_region_carriers_phasej_trainval_alpha1_policyval_pruned.json \
  --output_root /dev/shm/peilincai_spcarnet_vnext_counter_qual_20260626_125352 \
  --target_split test \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --method_name ours_26000_vnext_structure_aware_shrink \
  --wandb --wandb_mode offline \
  --wandb_group vnext_qual_counter_20260626 \
  --wandb_name vnext-qual-counter \
  --skip_teacher_cache \
  --strict_no_target_gt_apply \
  --texture_size 16 \
  --texture_size_candidates 16 \
  --support_expansion_mode none \
  --atlas_empty_bin_fill_mode face_mean \
  --surface_multiscale_prior_mode local_patch \
  --surface_multiscale_prior_blend_candidates 0.5 \
  --max_abs_delta_rgb_candidates 0.12 \
  --no_policy_val_bin_uncertainty_guard \
  --enable_policy_val_structure_aware_shrink \
  --structure_shrink_l1_weight 1.0 \
  --structure_shrink_gradient_weight 1.0 \
  --structure_shrink_edge_weight 0.0 \
  --structure_shrink_risk_tau 0.002 \
  --structure_shrink_max_penalty 1.0
```

Panel generation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/build_vnext_qualitative_panels.py \
  --gt_dir /dev/shm/peilincai_spcarnet_vnext_counter_qual_20260626_125352/counter/model/test/ours_26000_vnext_structure_aware_shrink/gt \
  --method cleanbest26000=outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/counter/test/ours_26000/renders \
  --method compact_base=outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model/test/ours_26000_phasef_extra_compact_base/renders \
  --method vnext=/dev/shm/peilincai_spcarnet_vnext_counter_qual_20260626_125352/counter/model/test/ours_26000_vnext_structure_aware_shrink/renders \
  --reference_label compact_base \
  --candidate_label vnext \
  --output_dir docs/car_model/vnext_artifacts/counter_qualitative_panel_20260626_125352 \
  --panel_name counter_cleanbest_base_vnext_panel \
  --num_views 6 \
  --tile_width 300 \
  --selection_mode largest_candidate_reference_delta
```

## Engineering Addition

New reusable tools:

```text
scripts/car_model/build_vnext_qualitative_panels.py
scripts/car_model/smoke_test_vnext_qualitative_panels.py
```

The exporter creates a PNG contact sheet, a manifest JSON, and a short summary Markdown. It can be reused for future full9 reruns that preserve render outputs instead of using cleanup-only packaging.

## Next Required Work

- Run a no-cleanup qualitative-preserving pass for the scenes where vNext has the clearest nonzero accepted outputs.
- Build panels only against the clean-best selected checkpoint, not arbitrary longer or worse clean checkpoints.
- Treat the current counter result as evidence that the pipeline is auditable, not as evidence that vNext has solved the visual-quality gap.
