# Surface Residual Region Texture Adapter Audit

- accepted: `True`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/room_teacher_surface_evidence_phasej_trainval_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/room/target_evidence_no_gt`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/room_teacher_render_visible_region_carriers_phasej_trainval_alpha1_policyval_pruned.json`
- support expansion mode: `none`
- support expansion base faces: `1160`
- support expansion added faces: `0`
- candidate faces after expansion: `1160`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `1160`
- fit samples: `1023700`
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
- surface multiscale prior blended bins: `17926`
- surface multiscale prior blended-bin fraction: `0.060365`
- surface multiscale prior mean blend weight: `0.090717`
- surface multiscale prior gate rejected bins: `233805`
- surface multiscale prior empty-bin rejects: `195417`
- surface multiscale prior sign rejects: `229037`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `42384`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `none`
- view-conditioned basis guard decision: `fallback_to_mean`
- view-conditioned basis supported bins: `18037`
- view-conditioned basis supported-bin fraction: `0.060739`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `245`
- teacher-distilled basis supported-face fraction: `0.211207`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `340299`
- selected alpha: `0.0625`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.829580`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `67040`
- local alpha fallback bins: `58848`
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `217`
- face gain guard rejected faces: `943`
- face gain guard allowed sample fraction: `0.545713`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.007586`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.001577`
- policy-val min-view relative gain: `0.000265`
- policy-val image SSIM gain: `0.000012413`
- policy-val image SSIM positive-view fraction: `0.916667`
- policy-val image SSIM min-view gain: `-0.000002623`
- policy-val image L1 gain: `0.000001152`
- policy-val image L1 positive-view fraction: `0.833333`
- policy-val image L1 min-view gain: `-0.000000767`
- policy-val risk gate: `True`
- target written views: `39`
- target changed fraction: `0.005199`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00461406 | 0.00461406 |
| 0.0020 | 0.000245 | 1.000000 | 0.000057 | 0.000019 | 0.000000447 | 0.916667 | -0.000000119 | 0.000000037 | 0.833333 | -0.000000024 | 0.00461406 | 0.00461293 |
| 0.0039 | 0.000490 | 1.000000 | 0.000114 | 0.000038 | 0.000000859 | 0.916667 | -0.000000179 | 0.000000074 | 0.833333 | -0.000000045 | 0.00461406 | 0.00461180 |
| 0.0078 | 0.000978 | 1.000000 | 0.000227 | 0.000073 | 0.000001689 | 0.916667 | -0.000000358 | 0.000000148 | 0.833333 | -0.000000091 | 0.00461406 | 0.00460955 |
| 0.0156 | 0.001947 | 1.000000 | 0.000445 | 0.000134 | 0.000003328 | 0.916667 | -0.000000656 | 0.000000296 | 0.833333 | -0.000000184 | 0.00461406 | 0.00460508 |
| 0.0312 | 0.003861 | 1.000000 | 0.000856 | 0.000223 | 0.000006492 | 0.916667 | -0.000001311 | 0.000000587 | 0.833333 | -0.000000373 | 0.00461406 | 0.00459625 |
| 0.0625 | 0.007586 | 1.000000 | 0.001577 | 0.000265 | 0.000012413 | 0.916667 | -0.000002623 | 0.000001152 | 0.833333 | -0.000000767 | 0.00461406 | 0.00457906 |
| 0.1250 | 0.014629 | 0.916667 | 0.002613 | -0.000196 | 0.000022406 | 0.833333 | -0.000005603 | 0.000002224 | 0.833333 | -0.000001593 | 0.00461406 | 0.00454656 |
| 0.2500 | 0.027088 | 0.916667 | 0.002215 | -0.003297 | 0.000035455 | 0.833333 | -0.000013947 | 0.000004163 | 0.833333 | -0.000003293 | 0.00461406 | 0.00448908 |
| 0.5000 | 0.045493 | 0.750000 | -0.008789 | -0.018208 | 0.000035360 | 0.750000 | -0.000082195 | 0.000007138 | 0.833333 | -0.000006886 | 0.00461406 | 0.00440416 |
