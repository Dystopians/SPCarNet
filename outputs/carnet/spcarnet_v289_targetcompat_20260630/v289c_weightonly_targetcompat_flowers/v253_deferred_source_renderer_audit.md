# v253 Deferred Source-Feature Residual Renderer

- verdict: `PASS_POLICY_VAL_PROMOTE_TO_FLOWERS_EXACT`
- policy-val all-axis pass: `True`
- selected alpha: `1.000000`
- no-target-GT audit pass: `True`
- target exact fixed-policy pass vs parent: `True`
- Phase-J flowers exact reference: `{'psnr': 20.304358, 'ssim': 0.55777, 'lpips': 0.329222}`

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_deferred_source_residual_renderer.py --bank_checkpoint /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz --drop_checkpoint_policy_fields --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented --target_eval_mode auto --policy_val_stride 4 --residual_decoder_mode view_feature_ridge_texture --grid 4 --min_source_count 2 --view_beta 3 --normal_beta 1 --parent_beta 8 --count_gamma 0.25 --gain_beta 1 --lowrank_basis_l2 0.05 --lowrank_basis_blend 0.5 --lowrank_basis_min_sources 3 --lowrank_basis_min_unique_views 2 --lowrank_basis_residual_clip 0.12 --lowrank_basis_disagreement_beta 2.0 --view_feature_ridge_self_error_beta 1.0 --view_feature_ridge_self_error_floor 0.5 --view_feature_ridge_holdout_beta 0.5 --view_feature_ridge_holdout_floor 0.75 --view_feature_ridge_holdout_min_sources 2 --patch_coherent_radius 1 --patch_coherent_bin_sigma 0.9 --patch_coherent_edge_beta 4 --source_consistency_mode off --policy_reliability_mode patch_perceptual_v1 --policy_reliability_alpha 1.0 --policy_reliability_min_count 8 --policy_reliability_min_positive_fraction 0.52 --policy_reliability_gain_scale 0.00025 --policy_gain_mode positive_soft --policy_gain_max 1.5 --policy_gain_scale 0.000025 --ood_gain_mode off --target_compatibility_mode soft --target_compatibility_view_sharpness 4.0 --target_compatibility_min_view_cos -0.2 --target_compatibility_beta 0.0 --target_compatibility_floor 1.0 --alpha_grid 1.0 --eval_chunk_size 8192 --compute_lpips --policy_val_ssim_max_side 512 --policy_val_lpips_max_side 256 --target_preview_views 2 --enable_wandb --wandb_project spcarnet-v169-target-compatibility --wandb_run_name v289c_weightonly_targetcompat_flowers --output_dir /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v289_targetcompat_20260630/v289c_weightonly_targetcompat_flowers --seed 291
```

## Method Change

v253 stores multiple train-fit teacher residual sources per face/UV bin, then uses a deferred renderer to mix them by target view direction, normal agreement, parent-RGB similarity, support count, and teacher-gain confidence. This changes the representation carrier rather than tuning an alpha gate.

`face_texture_lowrank` is the v169-oriented representation upgrade: it collects train-fit Phase-J teacher residual samples from a coherent UV neighborhood on the same mesh face, fits a compact RGB low-rank texture basis, and predicts target-view coefficients from view, parent appearance, edge, and relative-UV features. This is meant to test whether the teacher residual can be baked into a surface-attached representation instead of another scalar residual atlas.

`view_feature_ridge_texture` is a post-v282 carrier-capacity test: it keeps the same surface-attached face/UV source texture, but replaces the PCA residual bottleneck with a tiny weighted ridge decoder over view direction, normal, parent RGB, edge strength, teacher support, residual-edge support, source gain, support count, and relative-UV features. The target feature vector uses only target no-GT evidence plus train-fit source statistics, so it tests a stronger representation without target/test RGB leakage.

## Residual Decoder

- mode: `view_feature_ridge_texture`
- local-linear ridge: `0.050000`
- local-linear blend: `1.000000`
- local-linear min sources: `3`
- local-linear residual clip: `0.120000`
- lowrank basis rank: `3`
- lowrank basis min sources: `3`
- lowrank basis min unique views: `2`
- lowrank basis ridge: `0.050000`
- lowrank basis blend: `0.500000`
- lowrank basis residual clip: `0.120000`
- lowrank basis disagreement beta: `2.000000`
- view-feature ridge self-error beta: `1.000000`
- view-feature ridge self-error floor: `0.500000`
- view-feature ridge holdout beta: `0.500000`
- view-feature ridge holdout floor: `0.750000`
- view-feature ridge holdout min sources: `2`
- patch/texture radius: `1`
- patch/texture bin sigma: `0.900000`
- patch coherent blend: `0.250000`
- patch coherent min sources: `1.000000`
- patch coherent parent beta: `8.000000`
- patch coherent edge beta: `4.000000`
- patch coherent disagreement beta: `4.000000`
- patch coherent residual clip: `0.120000`
- source edge score weight: `0.000000`
- target edge gain: `0.000000`
- target edge gain clip: `1.500000`
- target compatibility mode: `soft`
- target compatibility view sharpness: `4.000000`
- target compatibility min view cosine: `-0.200000`
- target compatibility beta/floor: `0.000000` / `1.000000`
- target compatibility min effective sources: `2.000000`

`lowrank_source_basis` is a source-slot low-rank teacher-residual basis over the train-fit source bank. It checks independent source-view support through `source_view_id`, but it is not yet a coherent per-face texture sheet across UV bins. `face_texture_lowrank` is the explicit coherent face-texture variant introduced for the v169 prompt gate. `hybrid_edge_texture_lowrank` keeps the stable edge-local-linear base and injects that coherent face-texture basis as a controlled residual carrier. `structure_safe_texture_lowrank` adds a v274 structure certificate: it stores residual-edge, luma residual magnitude, and teacher-better support in the bank, then injects texture bases only when source/target edge evidence and multi-view support agree. `view_feature_ridge_texture` removes the low-rank coefficient bottleneck and directly decodes a view-dependent residual from the coherent surface texture features.

## Source-View Consistency

- mode: `off`
- valid source slots: `0`
- reliable source slots: `0`
- reliable fraction: `0.000000`
- mean reliability: `1.000000`
- mean amplitude: `1.000000`
- mean LOO relative error: `0.000000`
- mean LOO cosine: `0.000000`
- denoise enabled / source-slot fraction / energy ratio: `False` / `0.000000` / `1.000000`

## Policy-Val Metrics

| row | alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR tail CVaR | SSIM tail CVaR | LPIPS tail CVaR | active/support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best | 1.000000 | +0.048069 | +0.002057 | +0.000794 | +0.031795 | +0.001337 | +0.000276 | 0.877264 |
| best all-axis | 1.000000 | +0.048069 | +0.002057 | +0.000794 | +0.031795 | +0.001337 | +0.000276 | 0.877264 |

## Target Compatibility Diagnostics

| row | mean conf | p10 conf | mean risk | view gap | effective sources | unique sources |
|---|---:|---:|---:|---:|---:|---:|
| best | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| best all-axis | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Teacher Projection At Selected Alpha

| scope | cosine | energy retention | changed fraction | residual PSNR |
|---|---:|---:|---:|---:|
| full surface | 0.308105 | 0.157061 | 0.208570 | 31.825588 |
| active surface | 0.313890 | 0.163215 | 0.236190 | 31.416153 |

## Source Bank

- selected faces: `58023`
- selected score coverage: `n/a`
- nonempty face bins: `414433`
- nonempty source slots: `1308499`

## Policy Reliability

- mode: `patch_perceptual_v1`
- active bins: `57929`
- mean reliability: `0.025775`
- mean valid reliability: `0.170550`
- policy gain mode: `positive_soft`
- mean valid policy gain: `1.085081`
- max policy gain: `1.500000`
- mean valid tail risk: `0.799870`
- OOD gain mode: `off`

## Learned OOD/Gain Head

- mode: `off`
- sample count: `0`
- floor / ceiling: `0.000000` / `1.000000`
- label mean: `0.000000`
- predicted confidence mean: `0.000000`
- label/prediction correlation: `0.000000`

## Artifacts

- JSON: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v289_targetcompat_20260630/v289c_weightonly_targetcompat_flowers/v253_deferred_source_renderer_audit.json`
- Markdown: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v289_targetcompat_20260630/v289c_weightonly_targetcompat_flowers/v253_deferred_source_renderer_audit.md`
- checkpoint: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v289_targetcompat_20260630/v289c_weightonly_targetcompat_flowers/v253_deferred_source_renderer_bank.npz`
- policy-val renders: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v289_targetcompat_20260630/v289c_weightonly_targetcompat_flowers/policy_val_best`
- target no-GT preview: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v289_targetcompat_20260630/v289c_weightonly_targetcompat_flowers/target_no_gt_preview`

## Target Exact Fixed-Policy Evaluation

| PSNR | SSIM | LPIPS | PSNR gain | SSIM gain | LPIPS gain | changed fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 19.841839 | 0.620214 | 0.180080 | +0.009785 | +0.000303 | +0.000255 | 0.033008 |

- render dir: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v289_targetcompat_20260630/v289c_weightonly_targetcompat_flowers/target_exact_fixed_policy`
- Phase-J comparison: `{'reference': {'psnr': 20.304358, 'ssim': 0.55777, 'lpips': 0.329222}, 'candidate_psnr_minus_phasej': -0.46251888204912817, 'candidate_ssim_minus_phasej': 0.062443501832701975, 'phasej_lpips_minus_candidate': 0.14914191062055937, 'beats_phasej_all_axis_under_reported_metric_scale': False}`
