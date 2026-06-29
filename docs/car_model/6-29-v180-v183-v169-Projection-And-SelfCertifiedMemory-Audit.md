# v180-v183 v169 Projection And Self-Certified Memory Audit

Date: 2026-06-29

## Verdict

This is a real v169-oriented representation attempt, but it is not a paper-result success.

The v169 prompt was followed in the important way: no full9 run was launched before the flowers gate. The new representation produced the first tiny policy-val all-axis row, then failed to generalize to flowers test/exact. Therefore the correct status is:

```text
engineering/research-progress / flowers-exact-fail / do-not-promote-to-full9
```

## Step 0: Storage And v168 Status

Storage was unsafe throughout this run:

- `/data`: effectively full, about `2.9M` free at the end of this audit.
- `/dev/shm`: effectively full, about `171M` free at the end of this audit.
- `/tmp`: root filesystem has space, but user quota is at `100G*`, so Python `tempfile` can fail there.

The v168 direct-teacher low-copy exact run completed:

- root: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers`
- manifest: `COMPLETE`
- no-GT verifier: `passed=true`
- official flowers metrics: `19.832031 / 0.505779 / 0.405906`
- Phase-J flowers gate: `20.304358 / 0.557770 / 0.329222`

Conclusion: v168 fixed the low-copy protocol path but failed the quality gate by a large margin.

## Prior Projection Evidence

Direct Phase-J teacher residual is useful when applied as an oracle-like image correction on policy-val:

- parent policy-val: `20.606437 / 0.717526 / 0.153316`
- raw teacher residual: `21.077943 / 0.733263 / 0.143153`
- processed teacher residual: `21.519715 / 0.763420 / 0.135716`

The target signal exists. The bottleneck is projection into a baked surface carrier.

Recent projection attempts:

| version | mechanism | best policy-val result | all-axis |
|---|---|---:|---|
| v180 | differentiable per-face surface decoder, 128 faces | `+0.000002 / -0.000000 / +0.000001` | no |
| v181 | differentiable per-face surface decoder, 512 faces | `+0.000002 / -0.000000 / +0.000001` | no |
| v182 | KNN surface lumigraph memory | `+0.000841 / -0.000001 / -0.000022` | no |

These runs show that capacity alone is not enough: MLP projection preserves LPIPS only at no-op scale, while KNN memory carries more residual but hurts LPIPS.

## v183 Representation Change

New script:

```text
scripts/car_model/probe_self_certified_surface_residual_memory.py
```

Core idea:

1. Build a train-only surface residual memory from Phase-J teacher residuals.
2. Store residual samples per `(face_id, UV bin)` with view-direction keys.
3. Split train-fit views again into memory-fit and calibration views.
4. Use the calibration split to assign a reliability score per surface bin, without target/test GT.
5. Predict policy-val or target residual from view-direction KNN, reliability, agreement, and chroma shrink.
6. Apply to target no-GT evidence only after policy-val all-axis pass.

This is not another alpha-only variant. The new part is the cross-view self-certified surface memory and train-only reliability field.

## v183 Policy-Val Certificate

Main policy-val run:

```bash
TMPDIR=/dev/shm/peilincai_tmp_v183 \
WANDB_MODE=offline \
WANDB_DIR=/dev/shm/peilincai_wandb_v183 \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_v183 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/probe_self_certified_surface_residual_memory.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --output_dir /dev/shm/peilincai_spcarnet_v183_self_certified_memory \
  --max_faces 512 --texture_size 16 --k_per_bin 8 --knn_k 4 --tau 0.08 \
  --chroma_shrink 0.35 --agreement_power 1.0 --reliability_power 1.0 \
  --uncalibrated_reliability 0.15 \
  --max_memory_samples_per_view 65536 --max_calibration_samples_per_view 32768 \
  --alpha_grid 0,0.005,0.01,0.015625,0.03125,0.0625,0.125,0.25 \
  --compute_lpips --lpips_max_side 192 \
  --enable_wandb --wandb_project spcarnet_meshprior \
  --wandb_run_name v183-self-certified-memory-flowers-shm
```

W&B offline:

```text
/dev/shm/peilincai_spcarnet_v183_self_certified_memory/wandb/offline-run-20260628_233025-wexc95sq
```

Representation summary:

- selected faces: `512`
- memory keys: `57165`
- memory entries: `266327`
- calibrated keys: `21438`
- mean reliability: `0.026253`
- median reliability: `0.0`

Best all-axis policy-val row:

| alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR pos | SSIM pos | LPIPS pos |
|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | `+0.000049` | `+0.000000457` | `+0.000000481` | 0.917 | 0.583 | 0.667 |

This passes only as a very weak policy-val certificate. It is enough to justify one flowers exact attempt, not enough to claim quality.

## v183 Flowers Exact

Fixed-policy target apply command:

```bash
TMPDIR=/dev/shm/peilincai_tmp_v183 \
WANDB_MODE=offline \
WANDB_DIR=/dev/shm/peilincai_wandb_v183_exact \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_v183 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/probe_self_certified_surface_residual_memory.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --output_dir /dev/shm/peilincai_spcarnet_v183_self_certified_memory_exact \
  --max_faces 512 --texture_size 16 --k_per_bin 8 --knn_k 4 --tau 0.08 \
  --chroma_shrink 0.35 --agreement_power 1.0 --reliability_power 1.0 \
  --uncalibrated_reliability 0.15 \
  --max_memory_samples_per_view 65536 --max_calibration_samples_per_view 32768 \
  --alpha_grid 0,0.005,0.01,0.015625,0.03125,0.0625,0.125,0.25 \
  --compute_lpips --lpips_max_side 192 --skip_policy_val_renders \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --target_alpha 0.125 \
  --method_name ours_26000_v183_self_certified_memory_flowers \
  --enable_wandb --wandb_project spcarnet_meshprior \
  --wandb_run_name v183-self-certified-memory-flowers-exact
```

W&B offline:

```text
/dev/shm/peilincai_spcarnet_v183_self_certified_memory_exact/wandb/offline-run-20260628_233434-n9ic6snj
```

No-GT verifier:

- target evidence dir: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- forbidden keys absent: `true`
- `target_gt_visible_to_apply=false`
- `target_residual_visible_to_apply=false`

Official evaluator command:

```bash
TMPDIR=/dev/shm/peilincai_tmp_v183 \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_v183_eval \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/evaluate_render_split_metrics.py \
  --model_path /dev/shm/peilincai_spcarnet_v183_self_certified_memory_exact/flowers_exact_target_apply \
  --split test \
  --methods ours_26000_v183_self_certified_memory_flowers \
  --output /dev/shm/peilincai_spcarnet_v183_self_certified_memory_exact/v183_official_test_results.json \
  --per_view_output /dev/shm/peilincai_spcarnet_v183_self_certified_memory_exact/v183_official_test_per_view.json \
  --merge_model_results
```

Official flowers metrics:

| method | PSNR | SSIM | LPIPS | gate vs Phase-J |
|---|---:|---:|---:|---|
| Phase-J flowers | 20.304358 | 0.557770 | 0.329222 | reference |
| v168 direct teacher low-copy | 19.832031 | 0.505779 | 0.405906 | fail |
| v183 self-certified memory | 19.832029 | 0.505779 | 0.405907 | fail |

Delta v183 vs v168:

- PSNR: `-0.000002`
- SSIM: `+0.000000298`
- LPIPS: `-0.000000775` as gain, meaning LPIPS is worse by about `0.000000775`.

Delta v183 vs Phase-J:

- PSNR: `-0.472329`
- SSIM: `-0.051991`
- LPIPS: `+0.076685` worse.

## Target Alpha Diagnostic

Using the saved v183 target delta, nonzero alpha never achieved all-axis target improvement under full-resolution metric functions.

| alpha | PSNR gain | SSIM gain | LPIPS gain |
|---:|---:|---:|---:|
| 0.015625 | `+0.000002546` | `+0.000000022` | `-0.000000145` |
| 0.03125 | `+0.000004910` | `+0.000000060` | `-0.000000381` |
| 0.0625 | `+0.000009867` | `+0.000000093` | `-0.000000901` |
| 0.125 | `+0.000019497` | `+0.000000107` | `-0.000002395` |
| 0.25 | `+0.000038114` | `-0.000000130` | `-0.000006940` |
| 0.5 | `+0.000072942` | `-0.000001509` | `-0.000019460` |
| 1.0 | `+0.000132193` | `-0.000007318` | `-0.000046716` |

Conclusion: this is not just a bad alpha choice. The target residual carried by this representation is too weak and slightly perceptually harmful.

## Why Full9 Was Not Run

v169 explicitly forbids full9 promotion unless flowers exact beats Phase-J all-axis:

- required: PSNR `>20.304358`, SSIM `>0.557770`, LPIPS `<0.329222`
- observed: PSNR `19.832029`, SSIM `0.505779`, LPIPS `0.405907`

Full9 would therefore be expensive and misleading.

## Bottleneck

The reliable portion of the projected surface residual is too small:

- train-only calibration mean reliability is only `0.026253`;
- median reliability is `0.0`;
- target changed fraction is only about `0.00147`;
- increasing alpha gives more PSNR but worsens LPIPS and eventually SSIM.

This suggests that face/UV/bin-level residual memory still aliases view-dependent teacher corrections. The next method should not simply enlarge support or tune alpha. It needs a representation that predicts a coherent image-space correction while remaining surface-native, for example a train-only local neural texture with patch/context decoding and an explicit LPIPS/SSIM calibration objective, or a deferred surface feature decoder trained with held-out-view perceptual losses before any target apply.

## Current Completion Status

Final status: NOT COMPLETE.

Completed checklist items:

- real method change implemented: yes, `probe_self_certified_surface_residual_memory.py`;
- baseline/current/improved evidence: v168 and v183 exact are available, v180-v182 projection ablations are available;
- metrics and qualitative outputs: saved under `/dev/shm/peilincai_spcarnet_v183_self_certified_memory_exact`;
- commands/configs/errors documented: yes, in this file;
- final review honestly marks weaknesses: yes.

Unfinished item:

- v183 does not pass flowers exact against Phase-J, so full9 and paper-readiness remain blocked.

Exact next command if continuing this line:

```bash
TMPDIR=/dev/shm/peilincai_tmp_v184 CUDA_VISIBLE_DEVICES=1 PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_v184 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/probe_self_certified_surface_residual_memory.py --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence --output_dir /dev/shm/peilincai_spcarnet_v184_context_decoder_probe --max_faces 1024 --texture_size 32 --k_per_bin 8 --knn_k 4 --tau 0.05 --chroma_shrink 0.2 --agreement_power 2.0 --reliability_power 0.5 --uncalibrated_reliability 0.05 --compute_lpips --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented --target_alpha 0.0625 --method_name ours_26000_v184_context_decoder_probe_flowers
```
