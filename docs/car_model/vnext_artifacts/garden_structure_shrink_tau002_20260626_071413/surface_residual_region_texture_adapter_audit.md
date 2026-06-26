# Surface Residual Region Texture Adapter Audit

- accepted: `True`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_structure_shrink_garden_strict_20260626_071413_garden_structure_strict/garden/target_evidence_no_gt`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1_policyval_pruned.json`
- support expansion mode: `none`
- support expansion base faces: `319`
- support expansion added faces: `0`
- candidate faces after expansion: `319`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `319`
- fit samples: `201676`
- selected support mode: `base_carrier`
- selected support added faces: `0`
- selected texture size: `16`
- selected fill mode: `face_mean`
- selected max abs delta RGB: `0.120000`
- max abs delta RGB candidates: `[0.12]`
- policy candidate dominance pruning: `True`
- policy candidates planned before pruning: `1`
- policy candidates planned after pruning: `1`
- policy candidates executed: `1`
- policy candidate early-stop mode: `none`
- policy candidate early-stop skipped: `0`
- surface multiscale prior mode: `local_patch`
- surface multiscale prior selected blend: `0.500000`
- surface multiscale prior blend candidates: `[0.5]`
- surface multiscale prior gate mode: `evidence_consistent`
- surface multiscale prior block sizes: `[2, 4, 8]`
- surface multiscale prior blended bins: `8224`
- surface multiscale prior blended-bin fraction: `0.100705`
- surface multiscale prior mean blend weight: `0.088017`
- surface multiscale prior gate rejected bins: `65212`
- surface multiscale prior empty-bin rejects: `52208`
- surface multiscale prior sign rejects: `62947`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `14059`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `none`
- view-conditioned basis guard decision: `fallback_to_mean`
- view-conditioned basis supported bins: `3257`
- view-conditioned basis supported-bin fraction: `0.039883`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `53`
- teacher-distilled basis supported-face fraction: `0.166144`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `71583`
- selected alpha: `0.125`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.954476`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `19527`
- local alpha fallback bins: `11335`
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `56`
- face gain guard rejected faces: `263`
- face gain guard allowed sample fraction: `0.548277`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.010304`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.003582`
- policy-val min-view relative gain: `0.000120`
- policy-val image SSIM gain: `0.000002772`
- policy-val image SSIM positive-view fraction: `0.916667`
- policy-val image SSIM min-view gain: `-0.000000119`
- policy-val image L1 gain: `0.000000317`
- policy-val image L1 positive-view fraction: `0.833333`
- policy-val image L1 min-view gain: `-0.000000574`
- policy-val risk gate: `True`
- target written views: `24`
- target changed fraction: `0.002050`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00366984 | 0.00366984 |
| 0.0020 | 0.000182 | 1.000000 | 0.000078 | 0.000028 | 0.000000040 | 0.416667 | 0.000000000 | 0.000000005 | 0.750000 | -0.000000007 | 0.00366984 | 0.00366917 |
| 0.0039 | 0.000363 | 1.000000 | 0.000155 | 0.000054 | 0.000000104 | 0.666667 | 0.000000000 | 0.000000011 | 0.833333 | -0.000000015 | 0.00366984 | 0.00366851 |
| 0.0078 | 0.000723 | 1.000000 | 0.000308 | 0.000105 | 0.000000224 | 0.916667 | 0.000000000 | 0.000000021 | 0.833333 | -0.000000030 | 0.00366984 | 0.00366718 |
| 0.0156 | 0.001436 | 1.000000 | 0.000604 | 0.000196 | 0.000000407 | 0.916667 | 0.000000000 | 0.000000042 | 0.833333 | -0.000000065 | 0.00366984 | 0.00366457 |
| 0.0312 | 0.002829 | 1.000000 | 0.001163 | 0.000341 | 0.000000765 | 0.916667 | 0.000000000 | 0.000000084 | 0.833333 | -0.000000136 | 0.00366984 | 0.00365946 |
| 0.0625 | 0.005489 | 1.000000 | 0.002148 | 0.000474 | 0.000001505 | 0.916667 | 0.000000000 | 0.000000165 | 0.833333 | -0.000000279 | 0.00366984 | 0.00364969 |
| 0.1250 | 0.010304 | 1.000000 | 0.003582 | 0.000120 | 0.000002772 | 0.916667 | -0.000000119 | 0.000000317 | 0.833333 | -0.000000574 | 0.00366984 | 0.00363203 |
| 0.2500 | 0.017907 | 0.916667 | 0.004298 | -0.003074 | 0.000004644 | 0.916667 | -0.000000894 | 0.000000581 | 0.833333 | -0.000001213 | 0.00366984 | 0.00360412 |
| 0.5000 | 0.025015 | 0.833333 | -0.007667 | -0.019405 | 0.000005553 | 0.666667 | -0.000004530 | 0.000000914 | 0.750000 | -0.000002712 | 0.00366984 | 0.00357804 |
