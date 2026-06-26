# Surface Residual Region Texture Adapter Audit

- accepted: `True`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/kitchen/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_structure_shrink_kitchen_strict_20260626_1023/kitchen/target_evidence_no_gt`
- region carrier: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/carrier.json`
- support expansion mode: `none`
- support expansion base faces: `1315`
- support expansion added faces: `0`
- candidate faces after expansion: `1315`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `1315`
- fit samples: `696342`
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
- surface multiscale prior blended bins: `29952`
- surface multiscale prior blended-bin fraction: `0.088973`
- surface multiscale prior mean blend weight: `0.089746`
- surface multiscale prior gate rejected bins: `269156`
- surface multiscale prior empty-bin rejects: `218778`
- surface multiscale prior sign rejects: `259201`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `49263`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `9669`
- view-conditioned basis supported-bin fraction: `0.028722`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `147`
- teacher-distilled basis supported-face fraction: `0.111787`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `237698`
- selected alpha: `0.125`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.924661`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `75336`
- local alpha fallback bins: `67144`
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `152`
- face gain guard rejected faces: `1163`
- face gain guard allowed sample fraction: `0.387705`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.015490`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.006051`
- policy-val min-view relative gain: `0.004132`
- policy-val image SSIM gain: `0.000020564`
- policy-val image SSIM positive-view fraction: `1.000000`
- policy-val image SSIM min-view gain: `0.000001192`
- policy-val image L1 gain: `0.000002900`
- policy-val image L1 positive-view fraction: `1.000000`
- policy-val image L1 min-view gain: `0.000000287`
- policy-val risk gate: `True`
- target written views: `35`
- target changed fraction: `0.003550`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00668244 | 0.00668244 |
| 0.0020 | 0.000266 | 1.000000 | 0.000122 | 0.000104 | 0.000000467 | 1.000000 | 0.000000119 | 0.000000048 | 1.000000 | 0.000000007 | 0.00668244 | 0.00668066 |
| 0.0039 | 0.000532 | 1.000000 | 0.000244 | 0.000206 | 0.000000924 | 1.000000 | 0.000000238 | 0.000000096 | 1.000000 | 0.000000011 | 0.00668244 | 0.00667889 |
| 0.0078 | 0.001061 | 1.000000 | 0.000484 | 0.000408 | 0.000001838 | 1.000000 | 0.000000536 | 0.000000191 | 1.000000 | 0.000000022 | 0.00668244 | 0.00667535 |
| 0.0156 | 0.002110 | 1.000000 | 0.000953 | 0.000795 | 0.000003601 | 1.000000 | 0.000001013 | 0.000000382 | 1.000000 | 0.000000045 | 0.00668244 | 0.00666835 |
| 0.0312 | 0.004170 | 1.000000 | 0.001850 | 0.001511 | 0.000006924 | 1.000000 | 0.000001788 | 0.000000756 | 1.000000 | 0.000000086 | 0.00668244 | 0.00665458 |
| 0.0625 | 0.008141 | 1.000000 | 0.003476 | 0.002704 | 0.000012641 | 1.000000 | 0.000002682 | 0.000001490 | 1.000000 | 0.000000164 | 0.00668244 | 0.00662804 |
| 0.1250 | 0.015490 | 1.000000 | 0.006051 | 0.004132 | 0.000020564 | 1.000000 | 0.000001192 | 0.000002900 | 1.000000 | 0.000000287 | 0.00668244 | 0.00657893 |
| 0.2500 | 0.027817 | 1.000000 | 0.008535 | 0.003233 | 0.000022536 | 0.833333 | -0.000022769 | 0.000005458 | 1.000000 | 0.000000454 | 0.00668244 | 0.00649656 |
| 0.5000 | 0.043218 | 0.916667 | -0.000431 | -0.011949 | -0.000023782 | 0.333333 | -0.000126004 | 0.000009529 | 1.000000 | 0.000000456 | 0.00668244 | 0.00639364 |
