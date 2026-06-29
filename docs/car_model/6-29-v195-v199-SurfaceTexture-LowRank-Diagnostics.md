# SPCarNet v195-v199 Surface-Texture and Support-Aware Low-Rank Diagnostics

Date: 2026-06-29

This note records the first implementation pass driven by
`docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

The hard promotion gate from that prompt is unchanged:

| Reference | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Phase-J flowers gate | 20.304358 | 0.557770 | 0.329222 |

A run is not promotable unless it beats Phase-J flowers on all three axes:
higher PSNR, higher SSIM, and lower LPIPS. None of v195-v199 passes this gate.

## What Changed In Code

The train/apply pipeline now has two representation-level method families beyond
the original image-space U-Net path:

1. `surface_texture_mlp`
   - Implemented in `scripts/car_model/train_surface_conditioned_residual_unet.py`
     as `SurfaceTextureResidualMLP`.
   - Stores a trainable per-face/per-UV-bin surface feature texture.
   - Uses a small decoder to convert surface feature plus local render evidence
     into an RGB residual.
   - Supports target apply through
     `scripts/car_model/apply_surface_conditioned_residual_unet_checkpoint.py`.

2. `lowrank_surface_texture`
   - Implemented as `SupportAwareLowRankSurfaceTexture`.
   - Stores a small rank-K residual basis per selected face/UV bin.
   - Predicts mixture weights and confidence from local evidence and support
     statistics.
   - Adds a hard no-op gate for unknown or low-support rows, so target pixels
     without train support are not modified.
   - Logs support diagnostics during target apply:
     `mean_known_face_fraction`, `mean_active_support_fraction`,
     `mean_active_support_changed_fraction`, and
     `mean_inactive_support_changed_fraction`.

The selection path also gained no-GT target-visible capacity allocation:

- `--surface_target_visible_evidence_dir` points to target evidence that must
  pass `verify_target_no_gt`.
- Only geometry/render-side keys are allowed. Target RGB GT and target residuals
  remain forbidden during apply.
- Target-visible face counts are used only to allocate face capacity, not to fit
  residual values.
- This is transductive target-camera geometry usage. It is not RGB-GT leakage,
  but it must be disclosed and should not be described as target-view-blind.

After subagent review, the script also gained a small provenance fix:

- `--artifact_prefix` controls checkpoint/report names for future runs, so new
  runs no longer have to write stale `v184_*` artifact names.
- Standalone checkpoint apply can infer `lowrank_surface_texture` and
  `surface_texture_mlp` from checkpoint state keys if older checkpoints do not
  carry `args.model_type`.

## Validation Commands And Artifacts

All runs below were launched with W&B offline logging and `/dev/shm` outputs to
avoid the `/data` disk quota bottleneck.

| Run | Purpose | Output root | W&B run | Official result JSON |
| --- | --- | --- | --- | --- |
| v195 | surface texture MLP, teacher-only | `/dev/shm/peilincai_spcarnet_v195_flowers_surface_texture_teacheronly` | `/dev/shm/peilincai_spcarnet_v195_flowers_surface_texture_teacheronly/wandb/offline-run-20260629_023652-wniqm38z` | `v195_official_test_results.json` |
| v196 | surface texture MLP, train-fit GT-assisted diagnostic | `/dev/shm/peilincai_spcarnet_v196_flowers_surface_texture_gtassist` | `/dev/shm/peilincai_spcarnet_v196_flowers_surface_texture_gtassist/wandb/offline-run-20260629_024447-k7tffijf` | `v196_official_test_results.json` |
| v197 | support-aware low-rank, teacher-only | `/dev/shm/peilincai_spcarnet_v197_flowers_lowrank_teacheronly` | `/dev/shm/peilincai_spcarnet_v197_flowers_lowrank_teacheronly/wandb/offline-run-20260629_030113-w9oo4z85` | `v197_official_test_results.json` |
| v198 | support-aware low-rank, train-fit GT-assisted diagnostic | `/dev/shm/peilincai_spcarnet_v198_flowers_lowrank_gtassist` | `/dev/shm/peilincai_spcarnet_v198_flowers_lowrank_gtassist/wandb/offline-run-20260629_030113-o7d4owek` | `v198_official_test_results.json` |
| v199 | support-aware low-rank plus no-GT target-visible face priority | `/dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly` | `/dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly/wandb/offline-run-20260629_032243-f2osol7l` | `v199_official_test_results.json` |

Representative v199 command:

```bash
CUDA_VISIBLE_DEVICES=3 \
TMPDIR=/dev/shm/peilincai_tmp_v199_flowers_teacheronly \
WANDB_MODE=offline \
WANDB_DIR=/dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly/wandb \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/train_surface_conditioned_residual_unet.py \
  --model_type lowrank_surface_texture \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --surface_target_visible_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --scene_name flowers \
  --method_name ours_26000_v199_targetvisible_lowrank_teacheronly_flowers \
  --output_dir /dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly \
  --steps 3200 \
  --train_max_side 640 \
  --patch_size 320 \
  --surface_texture_size 8 \
  --surface_decoder_hidden 96 \
  --surface_decoder_layers 4 \
  --surface_face_max_unique 16384 \
  --surface_face_min_alpha 0.03 \
  --surface_face_min_residual_l1 0.0 \
  --lowrank_rank 4 \
  --lowrank_min_bin_support 8 \
  --lowrank_basis_init_std 0.01 \
  --max_delta 0.20 \
  --confidence_bias 0.0 \
  --teacher_l1_weight 1.0 \
  --teacher_ssim_weight 0.25 \
  --teacher_grad_weight 0.08 \
  --gt_l1_weight 0 \
  --gt_ssim_weight 0 \
  --gt_lpips_weight 0 \
  --gt_grad_weight 0 \
  --delta_l1_weight 0.00005 \
  --alpha_grid 0,0.03125,0.0625,0.125,0.25,0.5,0.75,1 \
  --eval_tile 512 \
  --eval_overlap 32 \
  --ssim_max_side 512 \
  --lpips_max_side 256 \
  --compute_lpips \
  --enable_wandb \
  --wandb_run_name v199-flowers-targetvisible-lowrank-teacheronly \
  --seed 199
```

Official v199 evaluator:

```bash
CUDA_VISIBLE_DEVICES=3 \
TMPDIR=/dev/shm/peilincai_tmp_v199_eval \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_populate_eval_gt_from_target_evidence.py \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --output_model /dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly/flowers_exact_target_apply \
  --split test \
  --method_name ours_26000_v199_targetvisible_lowrank_teacheronly_flowers \
  --audit_path /dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly/v199_eval_gt_population_audit.json \
  --force

CUDA_VISIBLE_DEVICES=3 \
TMPDIR=/dev/shm/peilincai_tmp_v199_eval \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/evaluate_render_split_metrics.py \
  -m /dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly/flowers_exact_target_apply \
  --split test \
  --methods ours_26000_v199_targetvisible_lowrank_teacheronly_flowers \
  --output /dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly/v199_official_test_results.json \
  --per_view_output /dev/shm/peilincai_spcarnet_v199_flowers_targetvisible_lowrank_teacheronly/v199_official_test_per_view.json \
  --merge_model_results
```

Full per-run configs are stored in each `v184_surface_conditioned_unet_report.json`.

## Official Flowers Exact Results

| Run | Model | Train GT used | Target GT used during apply | PSNR | SSIM | LPIPS | Verdict vs Phase-J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Phase-J gate | fixed reference | n/a | n/a | 20.304358 | 0.557770 | 0.329222 | target |
| v195 | surface_texture_mlp | no | no | 19.878033 | 0.509020 | 0.402998 | fail all axes |
| v196 | surface_texture_mlp | yes | no | 20.084991 | 0.523929 | 0.385202 | fail all axes |
| v197 | lowrank_surface_texture | no | no | 19.834993 | 0.505835 | 0.405083 | fail all axes |
| v198 | lowrank_surface_texture | yes | no | 19.833418 | 0.505749 | 0.404551 | fail all axes |
| v199 | lowrank + target-visible capacity | no | no | 19.835337 | 0.505801 | 0.404194 | fail all axes |

The best official run in this batch is still v196, but v196 used train-fit GT
losses as a diagnostic. It also remains below Phase-J on all three official
metrics.

## Policy-Val Results

| Run | PSNR gain | SSIM gain | LPIPS gain | Mean changed fraction | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| v195 | +0.053232 | +0.003875 | +0.000935 | 0.477074 | policy-val pass, but target transfer fails |
| v196 | +0.355694 | +0.029502 | +0.015891 | 0.869055 | strong policy-val overfit/GT-assisted diagnostic, target still fails |
| v197 | +0.002863 | +0.000099 | +0.000186 | 0.021791 | safe but too conservative |
| v198 | +0.000193 | +0.000106 | +0.000756 | 0.027818 | safe but too conservative even with train-fit GT |
| v199 | +0.002386 | +0.000075 | +0.000132 | 0.046047 | target-visible priority improves coverage, not quality |

Policy-val is not a sufficient promotion signal here. v195 and v196 show large
policy-val gains but weak official target transfer. v197-v199 pass policy-val
with tiny gains because the support-aware gate changes too little of the image.

## Support Diagnostics

| Run | Known face fraction | Active support fraction | Changed fraction | Inactive support changed |
| --- | ---: | ---: | ---: | ---: |
| v197 | 0.050104 | 0.029397 | 0.021687 | 0.000000 |
| v198 | 0.050104 | 0.029397 | 0.026522 | 0.000000 |
| v199 | 0.167715 | 0.105916 | 0.066904 | 0.000000 |

v199 proves the target-visible capacity allocator works mechanically: known
face coverage and active support both increase. However, the final metrics stay
near v197/v198. The bottleneck is therefore not only face capacity. The current
teacher residuals do not generalize strongly from source/support views to target
views under this low-rank surface representation.

## Fairness Audit

- v195, v197, and v199 are teacher-only with `train_fit_gt_weight_sum = 0.0`.
- v196 and v198 intentionally use train-fit GT losses and are documented as
  diagnostics, not clean method claims.
- The script defaults still include nonzero GT loss. Teacher-only claims require
  explicitly passing all GT weights as zero.
- All target apply steps passed the no-GT verifier.
- `surface_target_visible_evidence_dir` uses no target RGB GT and no target
  residuals. It uses target-visible geometry only for capacity allocation.
- This target-visible capacity step is transductive/test-camera-visible geometry
  usage. It is acceptable for a diagnostic but must be disclosed in any method
  claim.
- Eval GT is populated only after target renders are written.
- Policy-val all-axis pass means improvement over the parent render on held-out
  fit/policy-val views. It does not enforce the Phase-J flowers gate.
- Manual `--target_alpha` should not be selected after official target metrics;
  fair runs should use policy-val alpha or a predeclared alpha.
- No v195-v199 result is promoted to full9 or paper-ready status.

## Lessons Learned

1. A surface texture feature table plus decoder is not enough by itself. It can
   pass policy-val, but the same learned residual writes do not transfer to the
   held-out target views.
2. The hard support gate fixes the over-broad-write failure mode. It prevents
   inactive rows from changing target pixels, which is visible in the zero
   inactive-support change rate.
3. The same support gate also makes the method too conservative. v197/v198
   changed only about 2-3 percent of target pixels, which cannot create a strong
   visual or metric gain.
4. Target-visible capacity allocation is useful but insufficient. v199 raises
   active target support to about 10.6 percent and changed pixels to about 6.7
   percent, but still fails the Phase-J gate.
5. The remaining problem is cross-view residual generalization, not just memory
   size or face selection.
6. The v195-v199 family is a useful negative ablation for simple baked surface
   texture / low-rank residual distillation. It is not a paper-ready replacement
   for Phase-J.
7. The policy-val certificate is too weak for this prompt: it can approve a
   candidate that improves over the parent while remaining far below Phase-J.
8. A follow-up residual projection audit confirms the failure is already visible
   before target promotion. v199 retains only `0.015229` of policy-val teacher
   residual energy with cosine `0.039391`; v195 retains `0.068206` with cosine
   `0.112638`; even v196 GT-assisted has cosine only `0.138419`.

## Next Research Direction

The next attempt should not be another parameter-only scan or another wider
low-rank sweep. First run a teacher-residual projection audit with per-view and
per-region energy retention:

- raw residual: `Phase-J teacher render - parent render`;
- projected carrier residual: residual after the proposed surface/field carrier;
- final applied residual: residual actually written into target renders;
- report PSNR/SSIM/LPIPS tail views plus residual energy retention by region.

If projection already destroys structure, the carrier must be replaced with a
view-conditioned neural texture/decoder trained with patch, gradient, and
held-out source-view validation losses. If projection is healthy but final apply
fails, the bug is downstream in masking, confidence, clipping, or target-transfer
dilution.

A stronger follow-up should:

- keep the no-GT target-visible support allocator;
- keep the inactive-support no-op guarantee;
- replace per-row low-rank residual storage with a view-conditioned residual
  field that can model specular/view-dependent color and confidence;
- train it with explicit held-out source-view validation and reject rows/bases
  that do not improve held-out views;
- only run full9 after flowers exact beats Phase-J on PSNR, SSIM, and LPIPS.

Final status for this batch: NOT COMPLETE.

Follow-up projection audit:

```text
docs/car_model/6-29-v191-v199-ResidualProjectionAudit-Summary.md
docs/car_model/results/v191_v199_residual_projection_summary.json
```
