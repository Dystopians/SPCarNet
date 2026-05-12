# Phase-S Gain-Certified Face-Local SH1 Audit

Date: 2026-05-12

## Purpose

This note records the first Phase-S variant that converts the earlier face-local SH1 idea from a mostly consensus-filtered operator into a train-only certified policy.

The immediate goal was not to claim paper completion. The goal was to fix the specific outdoor failure observed in the prior probes: a face may have enough directional consensus, yet still be harmless on one train-val split and weak or unstable on another. The new variant adds a per-face gain certificate before topology refinement is allowed to modify the mesh.

## Method Change

Files changed:

- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

New policy component:

- Each selected face still needs train-only multi-view directional consensus.
- Each selected face is also evaluated on train-only policy-validation cameras.
- For each candidate face and local SH1 policy, the runner samples pixels from the face projection and estimates whether the predicted residual correction reduces residual MSE.
- A face is allowed only when enough train-only policy-validation views pass the gain test.
- Test views are never used for the gate.

New runner-facing options:

```text
--delta_min_face_gain_certificate_views
--delta_min_face_gain_certificate_relative_gain
--delta_min_face_gain_certificate_view_samples
--delta_min_face_gain_certificate_fraction
```

Fixed v1 policy:

```text
--delta_operator facelocal_sh1
--delta_min_face_view_consensus 0.67
--delta_min_face_consensus_views 2
--delta_min_face_consensus_view_samples 4
--delta_face_consensus_min_cosine 0.0
--delta_min_face_gain_certificate_views 2
--delta_min_face_gain_certificate_relative_gain 0.0
--delta_min_face_gain_certificate_view_samples 4
--delta_min_face_gain_certificate_fraction 0.67
```

## Commands

Environment:

```bash
export WANDB_MODE=online
export WANDB_PROJECT=mesh-splatting-ecsr
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
```

Single-scene gate command pattern:

```bash
$PY scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --scenes {garden|flowers|bicycle} \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16 \
  --iteration 26000 \
  --gpu {gpu_id} \
  --delta_operator facelocal_sh1 \
  --delta_uniform_barycentric \
  --evidence_max_views 16 \
  --evidence_view_stride 3 \
  --evidence_high_error_quantile 0.70 \
  --delta_top_k 16384 \
  --delta_min_view_hits 2 \
  --delta_min_consistency 0.65 \
  --delta_min_pixel_count 32 \
  --delta_strength 0.08 \
  --delta_max_abs_rgb 0.12 \
  --delta_max_faces_to_apply 4096 \
  --delta_min_policy_val_relative_gain 0.02 \
  --delta_min_policy_val_samples 512 \
  --delta_min_policy_val_unique_faces 16 \
  --delta_min_face_policy_val_relative_gain 0.0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 2 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0.0 \
  --delta_min_face_gain_certificate_views 2 \
  --delta_min_face_gain_certificate_relative_gain 0.0 \
  --delta_min_face_gain_certificate_view_samples 4 \
  --delta_min_face_gain_certificate_fraction 0.67 \
  --candidate_label facelocal_gaincert_v1_cached_dense16_20260512 \
  --candidate_base_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_base \
  --candidate_test_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela \
  --candidate_trainval_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_trainval_gate \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group phase_s_facelocal_gaincert_v1_cached_dense16_20260512 \
  --wandb_name phase_s_gaincert_v1_{scene}
```

Strict four-offset train-val command pattern:

```bash
$PY scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
  --scene {garden|flowers|bicycle} \
  --phasej_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/{scene}/ratio_0200/compact_model \
  --candidate_model outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/{scene}/model \
  --candidate_audit_json outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/{scene}/model/surface_residual_facelocal_sh1_delta_audit.json \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512 \
  --candidate_label facelocal_gaincert_v1_cached_dense16_20260512 \
  --candidate_base_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_base \
  --candidate_test_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela \
  --phasej_trainval_method_prefix ours_26000_phasej_gaincert_v1_20260512 \
  --candidate_trainval_method_prefix ours_26000_facelocal_gaincert_v1_20260512_multifold \
  --offsets 0,1,2,3 \
  --iteration 26000 \
  --gpu {gpu_id} \
  --policy_holdout_fraction 0.25 \
  --calib_sampler uniform \
  --calib_max_views 32 \
  --calib_stride 1 \
  --alpha_feature_mode confidence_magnitude_edge \
  --alpha_default 0.0 \
  --gate_min_psnr_gain 0.0 \
  --gate_max_ssim_regression 0.00005 \
  --gate_max_lpips_regression 0.00015 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group phase_s_facelocal_gaincert_v1_cached_dense16_20260512_multifold \
  --wandb_name {scene}_facelocal_gaincert_v1_cached_dense16_20260512_multifold
```

## Result Paths

Single-scene gates:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/garden/decisions/garden_decision.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/flowers/decisions/flowers_decision.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/bicycle/decisions/bicycle_decision.json`

Strict four-offset gates:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/garden/multifold_trainval_gate.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/flowers/multifold_trainval_gate.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/bicycle/multifold_trainval_gate.json`

Qualitative render outputs:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/garden/model/test/ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela/renders`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/flowers/model/test/ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela/renders`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/bicycle/model/test/ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela/renders`

## Single-Gate Results

All values below are candidate minus baseline. Selection uses train-val only; test is report-only.

| Scene | Gate | Train-val dPSNR | Train-val dSSIM | Train-val dLPIPS | Report-only test dPSNR | Report-only test dSSIM | Report-only test dLPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| garden | accept | +0.000175 | +0.000001 | -0.000002 | +0.000063 | +0.000001 | -0.000001 |
| flowers | accept | +0.000044 | +0.000001 | -0.000001 | +0.001677 | +0.000158 | -0.000305 |
| bicycle | reject | -0.000006 | +0.000000 | +0.000001 | +0.000374 | +0.000035 | -0.000115 |

## Strict Four-Offset Gate Results

| Scene | Strict gate | Mean dPSNR | Mean dSSIM | Mean dLPIPS | Main reason |
| --- | --- | ---: | ---: | ---: | --- |
| garden | accept | +0.000520 | +0.000017 | -0.000082 | all offsets pass |
| flowers | accept | +0.000030 | +0.000000 | +0.000000 | all offsets pass under v1 thresholds |
| bicycle | reject | +0.000143 | +0.000001 | +0.000023 | offset0 and offset2 PSNR below zero |

Offset-level details:

| Scene | Offset | dPSNR | dSSIM | dLPIPS | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| garden | 0 | +0.000175 | +0.000001 | -0.000002 | yes |
| garden | 1 | +0.001621 | +0.000066 | -0.000338 | yes |
| garden | 2 | +0.000221 | +0.000002 | +0.000013 | yes |
| garden | 3 | +0.000061 | +0.000000 | -0.000000 | yes |
| flowers | 0 | +0.000044 | +0.000001 | -0.000001 | yes |
| flowers | 1 | +0.000025 | +0.000000 | +0.000000 | yes |
| flowers | 2 | +0.000019 | +0.000000 | +0.000000 | yes |
| flowers | 3 | +0.000032 | +0.000001 | +0.000001 | yes |
| bicycle | 0 | -0.000006 | +0.000000 | +0.000001 | no |
| bicycle | 1 | +0.000006 | -0.000001 | -0.000000 | weak |
| bicycle | 2 | -0.000004 | -0.000000 | -0.000000 | no |
| bicycle | 3 | +0.000574 | +0.000006 | +0.000093 | no, LPIPS regression |

## Audit Counts

| Scene | Selected faces | Policy candidates | Accepted faces | Vertices added | Consensus passing | Gain-cert passing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| garden | 2578 | 156 | 156 | 468 | 290 / 1828 | 157 / 1828 |
| flowers | 2849 | 52 | 52 | 156 | 88 / 1425 | 53 / 1425 |
| bicycle | 3397 | 7 | 7 | 21 | 17 / 1256 | 7 / 1256 |

Topology integrity was clean in all three runs: no invalid faces, no degenerate faces, and no duplicate faces were reported in the audit files.

## Comparison Against Prior Phase-S Variants

The earlier face-local consensus v1 accepted garden but did not close the outdoor failure modes:

- `flowers` min2 fourfold rejected on offset3 because PSNR was below zero and LPIPS exceeded the regression cap.
- `flowers` min3 single-gate looked harmless on train-val but regressed on report-only test.
- `flowers` min3 strict fourfold still rejected on offset1.
- `bicycle` min2 fourfold rejected on offset2, and gaincert v1 still rejects bicycle on offset0 and offset2.

The gain certificate is therefore a real method improvement over the prior consensus-only policy for `flowers`, because it passes a fixed strict train-val four-offset gate that consensus-only did not pass. It is not yet a complete solution because `bicycle` remains rejected and the absolute improvements are still very small.

## W&B Runs

Single-gate:

- `garden`: test `h7qj5t00`, trainval `5ooytg02`
- `flowers`: test `xxni3lov`, trainval `6xl7jct7`
- `bicycle`: test `z8f3jiiq`, trainval `hbyjv1aa`

Strict fourfold:

- `garden`: `yuyutt2u`, `b0wcx9nz`, `3rxgpkx1`, `701qva6b`, `gvxfp8el`, `6pw7m713`, `c8n0ii71`, `kavekliy`
- `flowers`: `rsx2ez0e`, `1o72tbgh`, `esxhf0ht`, `1t70huvf`, `kw9rh6ep`, `sm9tiq6y`, `sr0mptt3`, `86bgt12q`
- `bicycle`: `qcrjdcg2`, `ilca5cfc`, `3p3ct5ty`, `v9wmu1xw`, `2gk6m8kq`, `zqy6hfkp`, `38mgez2k`, `erd4uypp`

## Errors And Blockers

No runtime error blocked the gaincert v1 experiments.

Scientific blockers remain:

- `bicycle` fails strict four-offset train-val acceptance.
- The successful scenes show low-amplitude image-metric gains; this is not yet a visually obvious paper-level result.
- The fixed policy has only been strict-gated on three scenes in this round, not on the complete scene set.
- The SP-CarNet multi-hypothesis branch still has a visible-surface regression versus K1, even though `rag_sym` improves reconstruction, hidden, and free-space metrics.

## Current Verdict

This is a milestone, not a closed loop.

Compared with the previous Phase-S face-local consensus policy, gaincert v1 fixes a concrete outdoor strict-gate failure on `flowers`. It does not yet deliver full baseline domination, full-scene validation, or visually obvious improvements. The next valid step is to turn this into a frozen policy ladder and run the remaining scenes, while separately attacking `bicycle` with a representation-level change rather than another parameter scan.
