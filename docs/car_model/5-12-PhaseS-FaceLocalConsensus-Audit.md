# Phase-S Face/View Consensus Surface-Code Audit

Date: 2026-05-12

## Purpose

Phase-R v11 established a strict train-only multi-offset gate, but accepted only
`3 / 9` representation-level edits.  The failures were not solved by residual
strength or gamma shrinking, so this phase tests a real operator change:
train-certified face/view ownership before writing surface residual SH1 codes.

The goal is to reduce the shared-vertex ownership error that hurt outdoor
scenes such as `garden`, `bicycle`, and `flowers`.  A face is allowed to carry a
residual code only when policy-validation train views agree on the residual
direction.  Held-out test metrics remain report-only.

## Method Change

Implemented interfaces:

- `scripts/car_model/ecsr_apply_surface_residual_barycentric_sh1_delta.py`
- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

New fixed-policy arguments:

- `--min_face_view_consensus`
- `--min_face_consensus_views`
- `--min_face_consensus_view_samples`
- `--face_consensus_min_cosine`

For each selected face, the operator groups policy-validation train samples by
view, computes a weighted residual RGB direction per view, and accepts the face
only if enough views agree with the mean face direction.  This certificate is
train-only and is recorded in the checkpoint audit under `face_view_consensus`.

Two carriers are tested:

1. `facelocal_sh1`: duplicates the three vertices of accepted faces and writes
   a local bounded SH1 residual code.
2. `sh1`: shared-vertex SH1 with the same consensus mask, used as an ablation.

## Acceptance Standard

The Phase-S candidate is not considered paper-closed unless it satisfies the
same v11 multi-offset train-only gate:

- offsets: `0,1,2,3`
- min PSNR gain: `0.0`
- max SSIM regression: `5e-5`
- max LPIPS regression: `1.5e-4`
- selection uses test: `false`

Target improvement over Phase-R v11:

- newly accept at least two hard fallback scenes, or reach at least `5 / 9`
  accepted representation edits;
- no accepted v11 scene regresses under the same multi-offset gate;
- report-only strict RGB wins increase beyond the current `3 / 9`.

## Commands Launched

Improved carrier, `garden`:

```bash
CUDA_VISIBLE_DEVICES=6 WANDB_MODE=online PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --scenes garden \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_v1_20260512 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_v1_20260512_surface_evidence \
  --iteration 26000 \
  --gpu 6 \
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
  --delta_min_face_policy_val_relative_gain 0.0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 2 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0.0 \
  --candidate_label facelocal_viewconsensus_v1_20260512 \
  --candidate_base_method ours_26000_facelocal_viewconsensus_v1_20260512_base \
  --candidate_test_method ours_26000_facelocal_viewconsensus_v1_20260512_phasej_ela \
  --candidate_trainval_method ours_26000_facelocal_viewconsensus_v1_20260512_trainval_gate \
  --wandb_group phase_s_facelocal_viewconsensus_v1_20260512 \
  --wandb_name garden_facelocal_viewconsensus_v1_20260512
```

Improved carrier, cached dense16 evidence probe, `garden`:

This uses an existing train-only dense evidence cache from Phase-R to avoid
waiting for the fresh-cache branch.  The output root and method names are new,
so rendered metrics are not reused.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online \
WANDB_TAGS=facelocal_viewconsensus_cached_dense16_20260512 \
PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --scenes garden \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_20260512 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16 \
  --iteration 26000 \
  --gpu 0 \
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
  --delta_min_face_policy_val_relative_gain 0.0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 2 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0.0 \
  --candidate_label facelocal_viewconsensus_cached_dense16_20260512 \
  --candidate_base_method ours_26000_facelocal_viewconsensus_cached_dense16_20260512_base \
  --candidate_test_method ours_26000_facelocal_viewconsensus_cached_dense16_20260512_phasej_ela \
  --candidate_trainval_method ours_26000_facelocal_viewconsensus_cached_dense16_20260512_trainval_gate \
  --wandb_group phase_s_facelocal_viewconsensus_cached_dense16_20260512 \
  --wandb_name garden_facelocal_viewconsensus_cached_dense16_20260512
```

Shared-vertex ablation, `garden`:

The first launch accidentally collided with an aborted worker using the same
output/evidence root:

- contaminated output root:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/shared_sh1_consensus_ablation_20260512`
- action: terminated the duplicate shared-SH1 processes and did not use that
  root for results

Clean relaunch:

```bash
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online \
WANDB_TAGS=shared_sh1_viewconsensus_ablation_v2_20260512 \
PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --scenes garden \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/shared_sh1_consensus_ablation_v2_20260512 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_s/surface_evidence_shared_sh1_consensus_ablation_v2_20260512 \
  --iteration 26000 \
  --gpu 7 \
  --delta_operator sh1 \
  --delta_uniform_barycentric \
  --delta_sh1_face_policy \
  --evidence_max_views 16 \
  --evidence_view_stride 3 \
  --evidence_high_error_quantile 0.70 \
  --delta_top_k 16384 \
  --delta_min_view_hits 2 \
  --delta_min_consistency 0.65 \
  --delta_min_pixel_count 32 \
  --delta_strength 0.08 \
  --delta_max_abs_rgb 0.012 \
  --delta_max_faces_to_apply 4096 \
  --delta_min_face_policy_val_relative_gain 0.0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 2 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0.0 \
  --candidate_label shared_sh1_viewconsensus_ablation_v2_20260512 \
  --candidate_base_method ours_26000_shared_sh1_viewconsensus_ablation_v2_20260512_base \
  --candidate_test_method ours_26000_shared_sh1_viewconsensus_ablation_v2_20260512_phasej_ela \
  --candidate_trainval_method ours_26000_shared_sh1_viewconsensus_ablation_v2_20260512_trainval_gate \
  --wandb_group phase_s_shared_sh1_viewconsensus_ablation_v2_20260512 \
  --wandb_name garden_shared_sh1_viewconsensus_ablation_v2_20260512
```

Existing accepted-control recheck, `kitchen` sparse SH1 v7 offset 0:

```bash
WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
  --scene kitchen \
  --phasej_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/kitchen/ratio_0200/compact_model \
  --candidate_model outputs/carnet/meshsplatopt/ecsr_phase_r/uniform_sh1_v7_sparse4096_indoor_kitchen/kitchen/model \
  --candidate_audit_json outputs/carnet/meshsplatopt/ecsr_phase_r/uniform_sh1_v7_sparse4096_indoor_kitchen/kitchen/model/surface_residual_barycentric_sh1_delta_audit.json \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_r/task4_worker/kitchen_sparse_sh1_v7_offset0_recheck_20260512 \
  --candidate_label uniform_sh1_v7_sparse4096_from_dense16_s012_top4096_task4_offset0 \
  --candidate_base_method ours_26000_uniform_sh1_v7_sparse4096_base \
  --candidate_test_method ours_26000_uniform_sh1_v7_sparse4096_phasej_ela \
  --phasej_trainval_method_prefix ours_26000_phasej_task4_kitchen_sparse_recheck \
  --candidate_trainval_method_prefix ours_26000_candidate_task4_kitchen_sparse_recheck \
  --offsets 0 \
  --gpu 7 \
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
  --wandb_group task4_worker_kitchen_sparse_recheck \
  --wandb_name task4_kitchen_sparse_sh1_v7_offset0_recheck
```

- output:
  `outputs/carnet/meshsplatopt/ecsr_phase_r/task4_worker/kitchen_sparse_sh1_v7_offset0_recheck_20260512/kitchen/multifold_trainval_gate.json`
- result: accepted on offset `0`
- train-val delta: PSNR `+0.001035690`, SSIM `+0.000033736`,
  LPIPS `-0.000005789`
- report-only test delta: PSNR `+0.022672653`, SSIM `+0.000718653`,
  LPIPS `-0.001068085`
- W&B base/current ELA:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/ht4740zr`
- W&B candidate ELA:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/fkn5c9mt`

## Current Status

The `garden` facelocal carrier passed the full four-offset train-only gate in
both cached-evidence and fresh-cache runs.  The `bicycle`/`flowers` outdoor
probe is running under the same fixed policy, and the shared-SH1 four-offset
ablation is running as the strict carrier-control.  Test metrics remain
report-only throughout.

Cached facelocal operator audit is available:

- audit:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_20260512/garden/model/surface_residual_facelocal_sh1_delta_audit.json`
- operator accepted: `true`
- no-op copy: `false`
- selected faces: `2578`
- face policy candidates: `224`
- accepted faces: `224`
- vertices added: `672`
- policy-val proxy relative gain: `0.185030`
- face/view consensus passing faces: `290 / 1828`
- coefficient abs mean / max: `0.00763220 / 0.06317358`

The candidate passed the four-offset train-val gate below.

Cached facelocal single train-val gate finished:

- decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_20260512/decisions/garden_decision.json`
- accepted: `true`
- selection uses test: `false`
- train-val delta: PSNR `+0.000213623`, SSIM `+0.000000834`,
  LPIPS `-0.000002325`
- report-only test delta: PSNR `+0.000082016`, SSIM `+0.000000775`,
  LPIPS `-0.000001654`
- W&B test ELA run:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/q7fjpf54`
- W&B train-gate ELA run:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/beb9796r`

Four-offset cached-evidence gate launched:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
  --scene garden \
  --phasej_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model \
  --candidate_model outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_20260512/garden/model \
  --candidate_audit_json outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_20260512/garden/model/surface_residual_facelocal_sh1_delta_audit.json \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_consensus_cached_dense16_20260512 \
  --candidate_label facelocal_viewconsensus_cached_dense16_20260512 \
  --candidate_base_method ours_26000_facelocal_viewconsensus_cached_dense16_20260512_base \
  --candidate_test_method ours_26000_facelocal_viewconsensus_cached_dense16_20260512_phasej_ela \
  --phasej_trainval_method_prefix ours_26000_phasej_phase_s_facelocal_cached_dense16_20260512 \
  --candidate_trainval_method_prefix ours_26000_facelocal_viewconsensus_cached_dense16_20260512_multifold \
  --offsets 0,1,2,3 \
  --gpu 0 \
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
  --wandb_group phase_s_facelocal_viewconsensus_cached_dense16_20260512_multifold \
  --wandb_name garden_facelocal_viewconsensus_cached_dense16_20260512_multifold
```

Four-offset cached-evidence gate finished:

- decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_consensus_cached_dense16_20260512/garden/multifold_trainval_gate.json`
- accepted: `true`
- selection uses test: `false`
- offsets accepted: `4 / 4`
- mean train-val delta: PSNR `+0.000572681`, SSIM `+0.000022039`,
  LPIPS `-0.000095792`

| offset | dPSNR | dSSIM | dLPIPS | decision |
|---:|---:|---:|---:|---|
| 0 | +0.000213623 | +0.000000834 | -0.000002325 | pass |
| 1 | +0.001615524 | +0.000065625 | -0.000337735 | pass |
| 2 | +0.000356674 | +0.000021636 | -0.000043869 | pass |
| 3 | +0.000104904 | +0.000000060 | +0.000000760 | pass |

Report-only held-out test delta: PSNR `+0.000082016`, SSIM
`+0.000000775`, LPIPS `-0.000001654`.

Interpretation: this is the first hard outdoor fallback scene from v11 that
passes the strict four-offset train-only representation gate.  The effect size
is still small, but the failure mode changed from "not robust under held-out
train folds" to "robust, low-amplitude representation repair."

Qualitative panel:

- panel:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/qualitative_panels/phase_s_garden_facelocal_20260512/garden_phasej_shared_facelocal_panel.png`
- manifest:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/qualitative_panels/phase_s_garden_facelocal_20260512/garden_phasej_shared_facelocal_panel_manifest.md`
- selection: top-4 held-out test views by Face-local vs Phase-J RGB MAE
  improvement
- columns: GT, Phase-J fallback, shared-SH1 ablation, face-local ours,
  magnified absolute difference

The selected held-out views improve RGB MAE only by `4e-7` to `3.2e-6`, so the
panel is useful as an audit artifact, not as a visually obvious paper figure.

Fresh-cache facelocal single train-val gate also finished:

- decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_v1_20260512/decisions/garden_decision.json`
- accepted: `true`
- selection uses test: `false`
- train-val delta: PSNR `+0.000213623`, SSIM `+0.000000834`,
  LPIPS `-0.000002325`
- report-only test delta: PSNR `+0.000082016`, SSIM `+0.000000775`,
  LPIPS `-0.000001609`

This matches the cached-evidence row closely, so the cached branch is not an
artifact of stale evidence reuse.

Fresh-cache facelocal four-offset gate finished:

- decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_v1_20260512/garden/multifold_trainval_gate.json`
- accepted: `true`
- selection uses test: `false`
- offsets accepted: `4 / 4`
- mean train-val delta: PSNR `+0.000572681`, SSIM `+0.000022039`,
  LPIPS `-0.000095826`
- report-only held-out test delta: PSNR `+0.000082016`, SSIM
  `+0.000000775`, LPIPS `-0.000001609`

This reproduces the cached-evidence four-offset row with a separately generated
evidence cache.  The matching deltas are expected because both runs materialize
nearly the same certified face-local edit, but the important audit point is
that neither path depends on test images for selection.

Shared-SH1 consensus ablation single train-val gate finished:

- decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/shared_sh1_consensus_ablation_v2_20260512/decisions/garden_decision.json`
- accepted: `true` under loose single-gate tolerances
- selection uses test: `false`
- train-val delta: PSNR `+0.000005722`, SSIM `-0.000017643`,
  LPIPS `+0.000090346`
- report-only test delta: PSNR `+0.000053406`, SSIM `+0.000000298`,
  LPIPS `-0.000000939`

The ablation is clearly weaker than face-local consensus: it passes only because
the single-gate tolerance allows small SSIM/LPIPS train-val regressions.  This
supports the method claim that local ownership is doing real work beyond a
shared-vertex consensus mask.

Shared-SH1 consensus ablation four-offset gate launched:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
  --scene garden \
  --phasej_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model \
  --candidate_model outputs/carnet/meshsplatopt/ecsr_phase_s/shared_sh1_consensus_ablation_v2_20260512/garden/model \
  --candidate_audit_json outputs/carnet/meshsplatopt/ecsr_phase_s/shared_sh1_consensus_ablation_v2_20260512/garden/model/surface_residual_barycentric_sh1_delta_audit.json \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/shared_sh1_consensus_ablation_v2_20260512 \
  --candidate_label shared_sh1_viewconsensus_ablation_v2_20260512 \
  --candidate_base_method ours_26000_shared_sh1_viewconsensus_ablation_v2_20260512_base \
  --candidate_test_method ours_26000_shared_sh1_viewconsensus_ablation_v2_20260512_phasej_ela \
  --phasej_trainval_method_prefix ours_26000_phasej_phase_s_shared_sh1_ablation_v2_20260512 \
  --candidate_trainval_method_prefix ours_26000_shared_sh1_viewconsensus_ablation_v2_20260512_multifold \
  --offsets 0,1,2,3 \
  --gpu 0 \
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
  --wandb_group phase_s_shared_sh1_consensus_ablation_v2_20260512_multifold \
  --wandb_name garden_shared_sh1_viewconsensus_ablation_v2_20260512_multifold
```

Shared-SH1 consensus ablation four-offset gate finished:

- decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/shared_sh1_consensus_ablation_v2_20260512/garden/multifold_trainval_gate.json`
- accepted: `false`
- selected fallback: `phasej_guarded_adaptedge`
- selection uses test: `false`
- failure reason: `offset1:psnr_gain_below_0`
- mean train-val delta: PSNR `+0.000010967`, SSIM `-0.000012904`,
  LPIPS `+0.000053991`
- report-only held-out test delta: PSNR `+0.000053406`, SSIM
  `+0.000000298`, LPIPS `-0.000000939`

| offset | dPSNR | dSSIM | dLPIPS | decision |
|---:|---:|---:|---:|---|
| 0 | +0.000005722 | -0.000017643 | +0.000090346 | pass under tolerance |
| 1 | -0.000223160 | -0.000034153 | +0.000079423 | fail |
| 2 | +0.000221252 | +0.000000477 | +0.000045523 | pass under tolerance |
| 3 | +0.000040054 | -0.000000298 | +0.000000671 | pass under tolerance |

This is the key carrier ablation.  The same train-only consensus idea is not
enough when the correction is written to shared vertices.  Face-local ownership
is the component that turns the operator from a single-gate weak positive into
a strict four-offset accepted edit on `garden`.

Fixed-policy outdoor probe, `bicycle` and `flowers`:

The first launch failed because `--scenes bicycle flowers` was passed as two
arguments.  The corrected launch used `--scenes bicycle,flowers`:

```bash
CUDA_VISIBLE_DEVICES=6 WANDB_MODE=online \
WANDB_TAGS=facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512 \
PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --scenes bicycle,flowers \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_outdoor_probe_20260512 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16 \
  --iteration 26000 \
  --gpu 6 \
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
  --delta_min_face_policy_val_relative_gain 0.0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 2 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0.0 \
  --candidate_label facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512 \
  --candidate_base_method ours_26000_facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512_base \
  --candidate_test_method ours_26000_facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512_phasej_ela \
  --candidate_trainval_method ours_26000_facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512_trainval_gate \
  --wandb_group phase_s_facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512 \
  --wandb_name outdoor_probe_bicycle_flowers_facelocal_viewconsensus_cached_dense16_20260512
```

`bicycle` single train-val gate finished:

- decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_outdoor_probe_20260512/decisions/bicycle_decision.json`
- accepted: `true`
- selection uses test: `false`
- train-val delta: PSNR `+0.000005722`, SSIM `-0.000000119`,
  LPIPS `-0.000000268`
- report-only held-out test delta: PSNR `+0.000370026`, SSIM
  `+0.000035286`, LPIPS `-0.000113904`
- operator audit: selected faces `3397`, face-policy candidates `14`,
  accepted faces `14`, vertices added `42`, face/view consensus passing
  `17 / 1256`

`flowers` single train-val gate finished:

- decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_outdoor_probe_20260512/decisions/flowers_decision.json`
- accepted: `true`
- selection uses test: `false`
- train-val delta: PSNR `+0.000038147`, SSIM `+0.000000596`,
  LPIPS `-0.000000924`
- report-only held-out test delta: PSNR `+0.001680374`, SSIM
  `+0.000158191`, LPIPS `-0.000305474`
- audit:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_outdoor_probe_20260512/flowers/model/surface_residual_facelocal_sh1_delta_audit.json`
- selected faces `2849`, face-policy candidates `70`, accepted faces `70`,
  vertices added `210`, face/view consensus passing `88 / 1425`

W&B links:

- bicycle test:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/4569aqel`
- bicycle train-val:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/jdkoledb`
- flowers test:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/bq0tgpsm`
- flowers train-val:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/65l6tmhj`

Interpretation: the same fixed policy now single-gate accepts all three v11
hard outdoor fallback scenes tested in Phase-S (`garden`, `bicycle`,
`flowers`).  Only `garden` has the full four-offset certificate so far; the
next fair claim requires four-offset validation for `bicycle` and `flowers`.

`bicycle` and `flowers` four-offset gates launched:

```bash
CUDA_VISIBLE_DEVICES=6 WANDB_MODE=online PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
  --scene bicycle \
  --phasej_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bicycle/ratio_0200/compact_model \
  --candidate_model outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_outdoor_probe_20260512/bicycle/model \
  --candidate_audit_json outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_outdoor_probe_20260512/bicycle/model/surface_residual_facelocal_sh1_delta_audit.json \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_consensus_cached_dense16_outdoor_probe_20260512 \
  --candidate_label facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512 \
  --candidate_base_method ours_26000_facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512_base \
  --candidate_test_method ours_26000_facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512_phasej_ela \
  --phasej_trainval_method_prefix ours_26000_phasej_phase_s_facelocal_outdoor_probe_20260512 \
  --candidate_trainval_method_prefix ours_26000_facelocal_viewconsensus_outdoor_probe_20260512_multifold \
  --offsets 0,1,2,3 \
  --gpu 6 \
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
  --wandb_group phase_s_facelocal_viewconsensus_outdoor_probe_20260512_multifold \
  --wandb_name bicycle_facelocal_viewconsensus_outdoor_probe_20260512_multifold
```

```bash
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
  --scene flowers \
  --phasej_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model \
  --candidate_model outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_outdoor_probe_20260512/flowers/model \
  --candidate_audit_json outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_cached_dense16_outdoor_probe_20260512/flowers/model/surface_residual_facelocal_sh1_delta_audit.json \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_consensus_cached_dense16_outdoor_probe_20260512 \
  --candidate_label facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512 \
  --candidate_base_method ours_26000_facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512_base \
  --candidate_test_method ours_26000_facelocal_viewconsensus_cached_dense16_outdoor_probe_20260512_phasej_ela \
  --phasej_trainval_method_prefix ours_26000_phasej_phase_s_facelocal_outdoor_probe_20260512 \
  --candidate_trainval_method_prefix ours_26000_facelocal_viewconsensus_outdoor_probe_20260512_multifold \
  --offsets 0,1,2,3 \
  --gpu 7 \
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
  --wandb_group phase_s_facelocal_viewconsensus_outdoor_probe_20260512_multifold \
  --wandb_name flowers_facelocal_viewconsensus_outdoor_probe_20260512_multifold
```

Status: running.  These decide whether the Phase-S carrier can upgrade v11 from
`3 / 9` accepted representation edits to a materially stronger fixed-policy
outdoor result.

Tail-risk refinement launched for `flowers`:

The first `flowers` four-offset run failed only on offset `3`, despite a strong
positive report-only test delta.  This suggests the current two-view consensus
certificate is not tail-safe enough.  A stricter fixed certificate was launched
that requires at least `3` policy-validation train views per accepted face:

```bash
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online \
WANDB_TAGS=facelocal_viewconsensus_min3_cached_dense16_20260512 \
PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --scenes flowers \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_consensus_min3_cached_dense16_20260512 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16 \
  --iteration 26000 \
  --gpu 7 \
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
  --delta_min_face_policy_val_relative_gain 0.0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 3 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0.0 \
  --candidate_label facelocal_viewconsensus_min3_cached_dense16_20260512 \
  --candidate_base_method ours_26000_facelocal_viewconsensus_min3_cached_dense16_20260512_base \
  --candidate_test_method ours_26000_facelocal_viewconsensus_min3_cached_dense16_20260512_phasej_ela \
  --candidate_trainval_method ours_26000_facelocal_viewconsensus_min3_cached_dense16_20260512_trainval_gate \
  --wandb_group phase_s_facelocal_viewconsensus_min3_cached_dense16_20260512 \
  --wandb_name flowers_facelocal_viewconsensus_min3_cached_dense16_20260512
```

## Paper Story

Phase-S reframes representation recovery as a surface ownership problem rather
than a parameter-strength problem.  The method does not simply add a larger
residual; it asks whether a surface element has stable train-view evidence that
it should own a local appearance correction.  Face-local SH1 then writes that
correction only onto local vertex copies, preventing one face's repair from
leaking through vertices shared with neighboring faces.

This is the right scientific direction if it works: it explains why v11
shared-vertex SH1 was reliable but tiny, and it gives a concrete mechanism for
turning train-only residual evidence into a persistent representation edit.

## Known Weaknesses

- The current experiments are still bounded to `garden` plus one accepted-scene
  control; full nine-scene closure is not yet complete.
- Face-local SH1 increases vertex count for accepted faces, so rate-distortion
  reporting must include vertex/attribute storage, not only triangle count.
- A positive single train-val gate is not enough; Phase-S must pass the
  four-offset gate before it can replace Phase-R v11.
- Test deltas are report-only and must not influence acceptance or fallback.

## 2026-05-12 Superseding Gain-Certificate Update

This consensus-only note is now an ablation record, not the latest method state.

The strict outdoor probe finished with mixed evidence:

- `bicycle` min2 fourfold rejected: `offset2:psnr_gain_below_0`.
- `flowers` min2 fourfold rejected: `offset3:psnr_gain_below_0` and `offset3:lpips_regression_exceeds_0.00015`.
- `flowers` min3 fourfold still rejected on offset1.

The follow-up method adds a train-only per-face gain certificate on top of face/view consensus. It is documented in:

- `docs/car_model/5-12-PhaseS-GainCertV1-Audit.md`

Gaincert v1 accepts `garden` and `flowers` under the strict four-offset train-val gate, and rejects `bicycle` under the same fixed policy. Therefore the method has made a real step beyond this consensus-only variant, but the paper loop is still not closed.
