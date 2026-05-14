# Phase-S Direct Patch-Cert Carrier Pilot

Date: 2026-05-14

This log records the completed 5-scene direct patch-certified carrier pilot.
It is paper-facing in structure, but the conclusion is intentionally strict:
the method change is real and auditable, yet it is not a broad Phase-S closure.

## Status

Artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_{garden,bicycle,counter,flowers,bonsai}
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_summary/summary_5scene_tail.md
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_qualitative/qualitative_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_qualitative/patchcert_qualitative_contact_sheet.png
```

Final decision source:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_{scene}/decisions/{scene}_decision.json
```

Completed scenes: `garden`, `bicycle`, `counter`, `flowers`, `bonsai`.
Accepted scenes: `1 / 5` (`bicycle`).

## Method Motivation

PatchRisk showed that a local carrier can be larger than a single residual
face, but it still expanded an already-selected plan. Direct patch-cert moves
carrier construction into the train-only certificate step:

1. Reuse dense train-only surface residual evidence.
2. Fit a bounded SH1 residual delta on the fixed Phase-J compact model.
3. Select seed faces by policy-val improvement and face/view consistency.
4. Expand each seed into a one-ring plus centroid-neighbor patch only when
   local support, view agreement, and residual direction agreement pass.
5. Apply patch shrink and fixed magnitude limits before materialization.
6. Promote through the Phase-K train-val gate with per-view tail constraints.
   Held-out test deltas are report-only.

This makes the carrier a direct output of the patch certificate rather than a
post-hoc replay of a previous candidate list.

## Fixed Command Template

Only `<scene>` and `<gpu>` vary by run. Numeric values are fixed across scenes.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --scenes <scene> \
  --gpu <gpu> \
  --skip_failed_views \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_<scene> \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16 \
  --candidate_label phase_s_patchcert_v5_patchcarrier_pilot_20260514 \
  --candidate_base_method ours_26000_phase_s_patchcert_v5_patchcarrier_pilot_20260514_base \
  --candidate_test_method ours_26000_phase_s_patchcert_v5_patchcarrier_pilot_20260514_phasej_ela \
  --candidate_trainval_method ours_26000_phase_s_patchcert_v5_patchcarrier_pilot_20260514_trainval_gate \
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
  --delta_min_face_policy_val_relative_gain 0.0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 2 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0.0 \
  --delta_min_face_gain_certificate_views 2 \
  --delta_min_face_gain_certificate_relative_gain 0.0 \
  --delta_min_face_gain_certificate_view_samples 4 \
  --delta_min_face_gain_certificate_fraction 0.75 \
  --delta_patch_cert_rings 1 \
  --delta_patch_cert_max_faces_per_seed 6 \
  --delta_patch_cert_min_direction_cosine 0.92 \
  --delta_patch_cert_min_neighbor_policy_val_samples 4 \
  --delta_patch_cert_min_neighbor_policy_val_relative_gain 0.0 \
  --delta_patch_cert_min_policy_val_samples 16 \
  --delta_patch_cert_min_relative_gain 0.02 \
  --delta_patch_cert_neighbor_mode both \
  --delta_patch_cert_centroid_candidates_per_seed 128 \
  --delta_patch_cert_shrink \
  --gate_min_psnr_gain 0 \
  --gate_max_ssim_regression 0.00005 \
  --gate_max_lpips_regression 0.00015 \
  --gate_min_balanced_delta 0 \
  --gate_tail_require_available \
  --gate_tail_cvar_fraction 0.2 \
  --gate_tail_max_balanced_negative_fraction 0.25 \
  --gate_tail_min_balanced_cvar_delta -0.0001 \
  --gate_tail_max_lpips_positive_fraction 0.35 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group phase_s_patchcert_v5_patchcarrier_pilot_20260514 \
  --wandb_name phase_s_patchcert_v5_patchcarrier_pilot_20260514_<scene>
```

Collector:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_collect_phasek_barycentric_gate_summary.py \
  --scenes garden,bicycle,counter,flowers,bonsai \
  --decision_path_template 'outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_{scene}/decisions/{scene}_decision.json' \
  --output_json outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_summary/summary_5scene_tail.json \
  --output_md outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_summary/summary_5scene_tail.md
```

Qualitative builder:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_build_phase_s_patchcert_qualitative.py \
  --scenes garden,bicycle,counter,flowers,bonsai \
  --views_per_scene 1 \
  --image_width 300 \
  --diff_boost 80 \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_qualitative
```

## Quantitative Summary

Selection uses train-val only. Test deltas are report-only. Effective deltas are
zero for rejected scenes because they fall back to Phase-J.

| scene | selected | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report-only test dPSNR | report-only test dSSIM | report-only test dLPIPS | effective dPSNR | effective dSSIM | effective dLPIPS | decision reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| garden | Phase-J fallback | false | +0.000118 | +0.000001 | -0.000001 | +0.000053 | +0.000000 | +0.000001 | +0.000000 | +0.000000 | +0.000000 | LPIPS tail-positive fraction too high |
| bicycle | direct patch-cert | true | +0.000021 | +0.000014 | -0.000026 | +0.000387 | +0.000036 | -0.000115 | +0.000387 | +0.000036 | -0.000115 | accepted; first hard-scene all-axis direct patch-cert win |
| counter | Phase-J fallback | false | +0.000174 | +0.000000 | -0.000001 | +0.000525 | -0.000015 | -0.000336 | +0.000000 | +0.000000 | +0.000000 | tail negative fraction, CVaR, and LPIPS tail fail |
| flowers | Phase-J fallback | false | +0.000065 | -0.000013 | +0.000004 | +0.005426 | +0.000471 | -0.000588 | +0.000000 | +0.000000 | +0.000000 | strong report-only positive, but train-val mean/tail fail |
| bonsai | Phase-J fallback | false | +0.000565 | -0.000003 | +0.000003 | -0.007896 | +0.000632 | +0.000819 | +0.000000 | +0.000000 | +0.000000 | tail fails and report-only PSNR/LPIPS regress |
| **mean** | - | **1/5** | - | - | - | - | - | - | **+0.000077** | **+0.000007** | **-0.000023** | sparse positive effective mean |

Tail-gate diagnostics:

| scene | train-val balanced delta | tail balanced CVaR | tail balanced negative fraction | LPIPS positive fraction | reading |
|---|---:|---:|---:|---:|---|
| garden | +0.000154912 | -0.000085652 | 0.219512 | 0.439024 | rejected by LPIPS tail |
| bicycle | +0.000819683 | -0.000075155 | 0.139535 | 0.162791 | accepted |
| counter | +0.000207543 | -0.000363675 | 0.509434 | 0.415094 | rejected by unstable tail |
| flowers | -0.000276089 | -0.001075260 | 0.657895 | 0.552632 | rejected despite strong report-only test |
| bonsai | +0.000456989 | -0.001601870 | 0.531250 | 0.515625 | rejected; report-only test confirms risk |

## Carrier Audit Snapshot

The operator materializes real checkpoint edits. It is not a no-op copy.

| scene | seed/policy faces | accepted patch faces | vertices added | note |
|---|---:|---:|---:|---|
| garden | 113 | 301 | 903 | non-noop edit, final gate rejects |
| bicycle | 7 | 30 | 90 | non-noop edit, final gate accepts |
| counter | 124 | 465 | 1395 | non-noop edit, final gate rejects |
| flowers | 37 | 109 | 327 | non-noop edit, report-only positive but train-val rejects |
| bonsai | 1249 | 2561 | 7683 | large non-noop edit, tail/test reject |

## Qualitative Evidence

Qualitative rows are selected by report-only held-out test balanced delta for
visualization only. They do not participate in selection.

Contact sheet:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_qualitative/patchcert_qualitative_contact_sheet.png
```

Summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_qualitative/qualitative_summary.md
```

Representative rows:

| scene | view | accepted | view dPSNR | view dSSIM | view dLPIPS | panel |
|---|---|---:|---:|---:|---:|---|
| bicycle | `00003.png` | true | +0.000122 | +0.000105 | -0.000312 | `.../bicycle_00003_patchcert_panel.png` |
| flowers | `00019.png` | false | +0.016310 | +0.001676 | -0.001602 | `.../flowers_00019_patchcert_panel.png` |
| bonsai | `00000.png` | false | +0.005825 | +0.001276 | -0.000869 | `.../bonsai_00000_patchcert_panel.png` |

The `flowers` and `bonsai` panels are useful for diagnosing train/test policy
mismatch, not for claiming accepted method gains.

## Comparison to PatchRisk

PatchRisk strict 5-scene replay:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_summary/summary_5scene_strict.md
```

It accepts `1 / 5` (`counter`) with mean effective deltas
`+0.000014877` PSNR, `+0.000000072` SSIM, and `-0.000000089` LPIPS.

Direct patch-cert accepts `1 / 5` (`bicycle`) with mean effective deltas
`+0.000077` PSNR, `+0.000007` SSIM, and `-0.000023` LPIPS. This is a stronger
hard-scene signal than PatchRisk, but it still does not solve broad coverage.

## Honest Weaknesses

- The accepted coverage is only `1 / 5`.
- The effective mean is positive but extremely small because four scenes fall
  back to Phase-J.
- `flowers` is the most important unresolved case: held-out test is strongly
  positive, but train-val aggregate and tail diagnostics reject it. Accepting it
  with the current policy would be test leakage.
- `bonsai` shows why tail-gating is necessary: train-val aggregate PSNR looks
  positive, but the tail is unstable and report-only PSNR/LPIPS regress.
- Carrier size is not free. `bonsai` adds `7683` vertices before rejection,
  proving that the operator needs a stronger capacity/risk predictor.
- Full-frame visual differences remain subtle; amplified error-delta panels
  are needed to inspect the local effect.

## Current Paper-Facing Reading

The defensible claim is:

> Direct patch-certified carrier construction is a real representation-level
> edit that can survive a strict train-val gate on a hard scene (`bicycle`), and
> its fair tail-gated replay exposes the exact policy bottleneck on `flowers`
> and `bonsai`. It is a meaningful mechanism milestone, not a final broad
> Phase-S endpoint.

Next work should redesign the train-only policy split/risk predictor rather
than search more fixed hyperparameter variants.
