# 6-29 v169 Surface U-Net Progress And Bottleneck Log

Date: 2026-06-29

This log records the v169-driven attempt after `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.
The short version is:

- `v191` is the first run in this line that passes the v169 flowers exact all-axis gate against the fixed Phase-J reference.
- `v192` improves counter strongly and even beats Phase-J on counter LPIPS, but it is still below Phase-J on counter PSNR and SSIM.
- `v193` adds face identity and a learned confidence head, but fails; it should be treated as negative evidence.
- A diagnostic target alpha sweep shows the counter gap is not solved by simple residual scaling.
- The current method is a surface-conditioned residual U-Net adapter, not yet a fully baked low-rank surface texture or paper-final method.

## Prompt Gate

The v169 prompt required that no large full9 promotion should happen before flowers exact beats Phase-J all-axis:

| reference | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-J flowers gate | 20.304358 | 0.557770 | 0.329222 |
| v191 flowers exact | 20.606058 | 0.578882 | 0.323687 |

Verdict: `v191` passes the flowers exact gate.

Main evidence:

- v191 flowers official results: `/dev/shm/peilincai_spcarnet_v191_lpips_grad_unet/v191_official_test_results.json`
- v191 flowers report: `/dev/shm/peilincai_spcarnet_v191_lpips_grad_unet/v184_surface_conditioned_unet_report.json`
- v191 no-GT/eval audit: `/dev/shm/peilincai_spcarnet_v191_lpips_grad_unet/v191_eval_gt_population_audit.json`
- local Phase-J flowers rerun: `/dev/shm/peilincai_spcarnet_v191_lpips_grad_unet/phasej_flowers_official_results_rerun.json`

The local Phase-J flowers rerun reports `20.300608 / 0.557458 / 0.329505`, which is slightly lower than the fixed prompt reference. For pass/fail, use the fixed v169 reference.

## Method Changes Implemented

### v191: Gradient/LPIPS-Aware Surface U-Net

Implemented in `scripts/car_model/train_surface_conditioned_residual_unet.py`.

The model takes parent render and surface buffers as input:

- parent RGB;
- normal;
- inverse depth;
- alpha;
- barycentric coordinates;
- texture channel;
- camera direction;
- valid surface mask.

It predicts a residual RGB field and applies:

```text
candidate = clamp(parent + alpha * predicted_residual)
```

Training objective was extended with:

- teacher residual L1/SSIM terms;
- train-fit GT L1/SSIM/LPIPS terms;
- luma-gradient preservation losses for teacher and GT targets;
- delta magnitude regularization.

Important honesty note: this is not pure teacher-only distillation. The successful v191/v192 runs used train-fit GT losses. Policy-val GT is also used for certification and alpha selection. Target/test GT is stripped before apply and only restored after apply for evaluation.

### v192: Counter-Oriented LPIPS/SSIM Reweighting

`v192` keeps the same representation but increases capacity and weights SSIM/LPIPS more heavily on counter.

It produced the best current counter result:

| method | PSNR | SSIM | LPIPS | comment |
|---|---:|---:|---:|---|
| clean/base counter | 26.751774 | 0.862055 | 0.252003 | local baseline from prior audit |
| v191 counter | 27.728537 | 0.884009 | 0.197185 | improves clean/base, below Phase-J |
| v192 counter | 28.097420 | 0.891432 | 0.184687 | best baked U-Net counter; LPIPS beats Phase-J |
| Phase-J counter | 28.449171 | 0.893731 | 0.186472 | strongest local endpoint |

Verdict: `v192` is useful but still not a counter Phase-J win because PSNR and SSIM are lower.

### v193: Face-Conditioned Confidence Head

New code added:

- optional `--confidence_mode sigmoid`;
- confidence head predicts per-pixel residual strength;
- optional face embedding path used as a surface identity feature;
- explicit GT-usage audit fields for future runs.

New utility added:

- `scripts/car_model/apply_surface_conditioned_residual_unet_checkpoint.py`

This utility applies a trained checkpoint to no-GT target evidence without retraining. It writes its own no-GT and GT-usage audit.

`v193` result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v193 counter | 27.584749 | 0.871107 | 0.227094 |

Verdict: reject. The confidence/face-embedding variant became too conservative and underfit useful correction.

## Counter Phase-J Gap Diagnosis

Using the true Phase-J counter endpoint in:

`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model/per_view.json`

and comparing to v192 counter:

| statistic | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| mean, positive is ours better | -0.351754 | -0.002299 | +0.001785 |
| min/tail worst | -1.922169 | -0.019316 | -0.025349 |
| positive-view fraction | 0.333333 | 0.366667 | 0.500000 |
| CVaR20 | -1.227091 | -0.016665 | -0.016729 |

Only `10 / 30` counter views are all-axis positive against Phase-J.

Worst views:

- worst PSNR: `00026.png`, `-1.922169 PSNR`, `-0.016669 SSIM`, `-0.011237 LPIPS gain`;
- worst SSIM: `00022.png`, `-0.553057 PSNR`, `-0.019316 SSIM`, `-0.014798 LPIPS gain`;
- worst LPIPS: `00018.png`, `-1.315008 PSNR`, `-0.018559 SSIM`, `-0.025349 LPIPS gain`.

Interpretation: the baked U-Net transfers a strong average residual, but Phase-J's view-adaptive endpoint is still sharper and more structurally aligned on several target views.

## Alpha Diagnostic

A checkpoint-apply interface was added to test whether counter is mostly an alpha calibration failure.
This is diagnostic only. It should not be used as the fair selected method because the target metric is examined after apply.

| run | alpha | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---:|---|
| v192 official fair alpha | 1.00 | 28.097420 | 0.891432 | 0.184687 | best fair v192 |
| v192 diagnostic | 0.75 | 28.115284 | 0.892193 | 0.188551 | PSNR/SSIM slight gain, LPIPS worse than Phase-J |
| v192 diagnostic | 1.25 | 27.963043 | 0.886453 | 0.188108 | worse |
| Phase-J counter | n/a | 28.449171 | 0.893731 | 0.186472 | endpoint reference |

Conclusion: residual scale calibration alone cannot close the counter all-axis gap. Lower alpha helps PSNR/SSIM slightly but loses LPIPS; higher alpha hurts all axes.

## No-GT And GT-Usage Audit

Target/test apply remained no-GT:

- v191 flowers apply: verifier passed, target/test GT was not visible during apply.
- v191 counter apply: verifier passed, target/test GT was not visible during apply.
- v192 counter apply: verifier passed, target/test GT was not visible during apply.
- v193 counter apply: verifier passed, target/test GT was not visible during apply.
- v192 alpha diagnostics: verifier passed, target/test GT was not visible during apply.

However, the successful training runs did use train-fit GT losses and policy-val GT:

```text
uses_train_fit_gt = true
uses_policy_val_gt = true
uses_target_or_test_gt_during_apply = false
```

This must be disclosed in slides/paper notes. A stricter teacher-only ablation is still needed before making a pure teacher-distillation claim.

## Commands And Artifact Index

Representative commands:

```bash
# v191 flowers mid run, W&B offline enabled
TMPDIR=/dev/shm/peilincai_tmp_v191_lpips_grad_unet \
WANDB_MODE=offline \
WANDB_DIR=/dev/shm/peilincai_spcarnet_v191_lpips_grad_unet/wandb \
CUDA_VISIBLE_DEVICES=2 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_surface_conditioned_residual_unet.py \
--fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
--target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
--target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
--steps 3200 --train_max_side 640 --patch_size 320 --base_channels 32 \
--teacher_grad_weight 0.03 --gt_lpips_weight 0.18 --gt_grad_weight 0.12 \
--compute_lpips --enable_wandb

# v192 counter mid run, W&B offline enabled
TMPDIR=/dev/shm/peilincai_tmp_v192_counter_lpips_ssim_unet \
WANDB_MODE=offline \
WANDB_DIR=/dev/shm/peilincai_spcarnet_v192_counter_lpips_ssim_unet/wandb \
CUDA_VISIBLE_DEVICES=3 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_surface_conditioned_residual_unet.py \
--fit_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/teacher_surface_evidence \
--target_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/target_evidence_no_gt \
--target_eval_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/target_evidence_reparented \
--steps 4200 --train_max_side 704 --patch_size 352 --base_channels 40 \
--gt_ssim_weight 0.30 --gt_lpips_weight 0.32 --gt_grad_weight 0.08 \
--compute_lpips --enable_wandb

# v193 rejected face-confidence run
TMPDIR=/dev/shm/peilincai_tmp_v193_counter_faceconf_unet \
WANDB_MODE=offline \
WANDB_DIR=/dev/shm/peilincai_spcarnet_v193_counter_faceconf_unet/wandb \
CUDA_VISIBLE_DEVICES=2 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_surface_conditioned_residual_unet.py \
--fit_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/teacher_surface_evidence \
--target_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/target_evidence_no_gt \
--confidence_mode sigmoid --confidence_min 0.05 --confidence_max 1.15 \
--face_embedding_dim 4 --steps 3600 --compute_lpips --enable_wandb

# checkpoint alpha diagnostic interface
CUDA_VISIBLE_DEVICES=2 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_surface_conditioned_residual_unet_checkpoint.py \
--checkpoint /dev/shm/peilincai_spcarnet_v192_counter_lpips_ssim_unet/v184_surface_conditioned_unet.pt \
--target_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/target_evidence_no_gt \
--output_model /dev/shm/peilincai_spcarnet_v192_counter_alpha075_diag/counter_exact_target_apply \
--method_name ours_26000_v192_alpha075_diag_counter \
--alpha 0.75
```

Full artifact paths:

- v191 flowers W&B: `/dev/shm/peilincai_spcarnet_v191_lpips_grad_unet/wandb/offline-run-20260629_010111-fwr5st62`
- v192 counter W&B: `/dev/shm/peilincai_spcarnet_v192_counter_lpips_ssim_unet/wandb/offline-run-20260629_012809-e9vx3ftd`
- v193 counter W&B: `/dev/shm/peilincai_spcarnet_v193_counter_faceconf_unet/wandb/offline-run-20260629_014235-4fmoc9bw`
- v191 flowers qualitative panel: `assets/spcarnet_v191_phasej_flowers_qualitative_panel.png`
- v191 flowers qualitative manifest: `assets/spcarnet_v191_phasej_flowers_qualitative_panel_manifest.json`

## Current Status

`v169` has produced a meaningful method improvement and a real flowers all-axis win over the fixed Phase-J gate, but it has not produced a paper-final closed loop.

Completed:

- real train/eval pipeline method change;
- W&B-logged medium runs;
- v191 flowers exact pass;
- v191/v192/v193 counter official evaluations;
- no-GT apply audits;
- checkpoint apply interface;
- diagnostic evidence that simple alpha calibration cannot close counter.

Still incomplete:

- counter does not beat Phase-J all-axis;
- full9 should not be promoted yet as a final claim;
- teacher signal diagnostics and carrier upper-bound reports are still incomplete;
- stricter teacher-only ablation is needed;
- the representation is still render-space U-Net conditioned on surface buffers, not the full low-rank/baked surface texture requested by v169.

Next technical direction:

1. Keep v192 as the strongest current counter baked U-Net baseline.
2. Implement a stricter teacher-only ablation to measure how much train-fit GT is carrying the success.
3. Replace pure render-space U-Net with an explicit surface-attached feature texture or low-rank per-face basis and only use the U-Net/MLP as a small decoder.
4. Add a robust policy-val gate that compares against Phase-J per-view tails, not only against the parent.
5. Do not claim full paper closure until counter-like scenes beat Phase-J on PSNR, SSIM, and LPIPS simultaneously.

Final status: NOT COMPLETE.
