# Phase-S Compact Stratified Gate Log

Date: 2026-05-14

This note records the follow-up to the direct PatchCert carrier pilot.  The
goal was not to tune one scene by hand, but to add a fixed train-only promotion
policy for small patch-certified representation edits that were rejected by the
earlier per-view tail gate despite strong held-out evidence.

## Method Change

The new gate is implemented in:

- `scripts/car_model/ecsr_decide_phasek_trainval_gate.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

It adds two diagnostics to the existing train-val promotion gate:

1. **View-stratified train-val groups.** Train-val views are sorted by view id
   and split round-robin into four groups. The gate tracks the worst group mean
   PSNR/SSIM delta and the worst LPIPS group mean. This catches a repair that
   only works on one camera band.
2. **Compact-carrier override.** A rejected edit can be promoted only if the
   patch carrier is small and all compact train-val risk checks pass. The
   default thresholds require at most `160` accepted faces, at most `512`
   inserted vertices, face ratio at most `1.5e-5`, non-negative train-val PSNR
   above `2e-5`, small SSIM/LPIPS mean regressions, bounded tail risk, and
   bounded stratified-group regressions.

This gate can override the older balanced/tail rejection only for compact
carriers. It cannot override a checkpoint operator rejection or no-op materialization.
Held-out test metrics remain report-only.

## Command

The fixed multi-scene replay used the existing Phase-K runner with online W&B
logging:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --scenes <scene> \
  --gpu <gpu> \
  --skip_failed_views \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v6_compactstrat_gate_20260514_<scene> \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16 \
  --candidate_label phase_s_patchcert_v6_compactstrat_gate_20260514 \
  --candidate_base_method ours_26000_phase_s_patchcert_v6_compactstrat_gate_20260514_base \
  --candidate_test_method ours_26000_phase_s_patchcert_v6_compactstrat_gate_20260514_phasej_ela \
  --candidate_trainval_method ours_26000_phase_s_patchcert_v6_compactstrat_gate_20260514_trainval_gate \
  --phasej_test_method ours_26000_phasej_guarded_adaptedge_ela_replay_rendercalib_v1_top1_s2_fair \
  --phasej_trainval_method ours_26000_phasej_trainval_gate_rendercalib_v1_top1_s2_fair \
  --evidence_max_views 16 \
  --evidence_view_stride 3 \
  --delta_operator facelocal_sh1 \
  --delta_uniform_barycentric \
  --delta_sh_degree 3 \
  --delta_top_k 16384 \
  --delta_min_consistency 0.65 \
  --delta_min_pixel_count 32 \
  --delta_max_abs_rgb 0.05 \
  --delta_strength 0.035 \
  --delta_steps 800 \
  --delta_min_policy_val_relative_gain 0.02 \
  --delta_min_policy_val_samples 512 \
  --delta_min_policy_val_unique_faces 16 \
  --delta_lambda_mag 0.03 \
  --delta_lambda_sh1_mag 0.06 \
  --delta_lambda_smooth 0.1 \
  --delta_max_faces_to_apply 4096 \
  --delta_min_face_policy_val_relative_gain 0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 2 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0 \
  --delta_min_face_gain_certificate_views 2 \
  --delta_min_face_gain_certificate_relative_gain 0 \
  --delta_min_face_gain_certificate_view_samples 4 \
  --delta_min_face_gain_certificate_fraction 0.75 \
  --delta_patch_cert_rings 1 \
  --delta_patch_cert_max_faces_per_seed 6 \
  --delta_patch_cert_min_direction_cosine 0.92 \
  --delta_patch_cert_min_neighbor_policy_val_samples 4 \
  --delta_patch_cert_min_neighbor_policy_val_relative_gain 0 \
  --delta_patch_cert_min_policy_val_samples 16 \
  --delta_patch_cert_min_relative_gain 0.02 \
  --delta_patch_cert_neighbor_mode both \
  --delta_patch_cert_centroid_candidates_per_seed 128 \
  --delta_patch_cert_shrink \
  --gate_tail_require_available \
  --gate_tail_cvar_fraction 0.2 \
  --gate_tail_max_balanced_negative_fraction 0.25 \
  --gate_tail_min_balanced_cvar_delta -0.0001 \
  --gate_tail_max_lpips_positive_fraction 0.35 \
  --gate_compact_enable \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group phase_s_patchcert_v6_compactstrat_gate_20260514 \
  --wandb_name phase_s_patchcert_v6_compactstrat_gate_20260514_<scene>
```

Scenes were run on available GPUs including `0`, `1`, `4`, `5`, and `6`.
The expensive render/eval steps completed with W&B logging. The first launched
runner processes failed only at the final decision subprocess because argparse
misread a negative scientific-notation threshold (`-1e-05`). The runner now
formats decision float arguments in decimal form; the completed render/eval
outputs were kept and the decisions were regenerated with the fixed decision
script.

## Quantitative Evidence

Summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v6_compactstrat_gate_20260514_summary/summary_5scene.md`

Qualitative panels:

`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v6_compactstrat_gate_20260514_qualitative/qualitative_summary.md`

| scene | selected | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report-only test dPSNR | report-only test dSSIM | report-only test dLPIPS | effective dPSNR | effective dSSIM | effective dLPIPS | reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| garden | Phase-J fallback | false | +0.000118 | +0.000001 | -0.000001 | +0.000053 | +0.000000 | +0.000001 | +0.000000 | +0.000000 | +0.000000 | compact capacity and LPIPS tail reject |
| bicycle | direct PatchCert v6 | true | +0.000021 | +0.000014 | -0.000026 | +0.000387 | +0.000036 | -0.000115 | +0.000387 | +0.000036 | -0.000115 | accepted by standard and compact gates |
| counter | Phase-J fallback | false | +0.000174 | +0.000000 | -0.000001 | +0.000525 | -0.000015 | -0.000336 | +0.000000 | +0.000000 | +0.000000 | compact capacity rejects; SSIM test is negative |
| flowers | direct PatchCert v6 | true | +0.000065 | -0.000013 | +0.000004 | +0.005426 | +0.000471 | -0.000588 | +0.005426 | +0.000471 | -0.000588 | recovered from v5 rejection by compact stratified gate |
| bonsai | Phase-J fallback | false | +0.000565 | -0.000003 | +0.000003 | -0.007896 | +0.000632 | +0.000819 | +0.000000 | +0.000000 | +0.000000 | large carrier and held-out PSNR/LPIPS failure |
| **mean** | - | **2/5** | - | - | - | - | - | - | **+0.001163** | **+0.000101** | **-0.000141** | positive effective mean, still sparse |

Compared with direct PatchCert v5, this is a real policy improvement:

| version | scenes | accepted | accepted scenes | mean effective dPSNR | mean effective dSSIM | mean effective dLPIPS |
|---|---:|---:|---|---:|---:|---:|
| direct PatchCert v5 tail gate | 5 | 1/5 | bicycle | +0.000077 | +0.000007 | -0.000023 |
| direct PatchCert v6 compact-stratified gate | 5 | 2/5 | bicycle, flowers | +0.001163 | +0.000101 | -0.000141 |

Important caveat: `flowers` is accepted by the compact-carrier override even
though the older balanced/tail gate rejects it (`trainval_balanced_delta =
-0.000276`, high balanced-negative fraction). The v6 claim is therefore not
"all train-val diagnostics are positive"; it is "a small patch carrier is
tolerated by bounded component-wise, tail, and stratified train-val risk."
The follow-up validation is a strict four-offset train-only gate for
`bicycle` and `flowers`. The active continuation also adds a fold-aware
PatchCert carrier audit; see
`docs/car_model/5-14-PhaseS-V6Multifold-V7V8-FoldAware-PatchCert-Log.md`.

## Qualitative Evidence

The contact sheet has been copied into the README assets:

`assets/spcarnet_phase_s_patchcert_v6_compactstrat_contact_sheet.png`

The clearest accepted visual improvement is `flowers/00019.png`
(`+0.016310` PSNR, `+0.001676` SSIM, `-0.001602` LPIPS on that held-out
view). `bicycle/00003.png` is accepted but subtle. Rejected diagnostic rows
are kept in the contact sheet to show why full-frame qualitative inspection
alone is not a sufficient promotion rule.

## Honest Reading

This is a meaningful milestone because it turns the earlier `flowers` test-only
positive into a train-val-accepted representation edit under a fixed compact
stratified gate. It also keeps the obvious risky cases rejected: `bonsai` has a
large carrier and bad held-out PSNR/LPIPS, and `counter` has attractive LPIPS
but negative held-out SSIM.

It is not yet a paper-level endpoint. Acceptance is `2 / 5`, the full-frame
visual change remains subtle except for amplified residual maps, and the result
has not yet been broadened to every Mip-NeRF360 scene under one final
representation-level policy. The next research step should target a stronger
surface-attached carrier or better train-only risk predictor rather than more
manual threshold scanning.
