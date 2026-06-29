# v184-v188 v169 Surface Feature Distillation Audit

Date: 2026-06-29

This log follows `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.
The hard gate is flowers exact against Phase-J:

- Phase-J flowers reference: PSNR `20.304358`, SSIM `0.557770`, LPIPS `0.329222`
- Full9 is forbidden until flowers exact beats all three axes.
- Target/test apply must use stripped no-GT evidence only.

## Storage And Runtime Preflight

At launch time:

- `/data`: full, about `2.7M` free.
- `/dev/shm`: about `15-17G` free during the run.
- `/tmp`: root filesystem quota near 100G, so all experiments used `/dev/shm`.
- W&B was enabled in offline mode for medium runs.

No duplicate full/exact evidence cache was launched. All target applies used:

- train-fit teacher evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- target no-GT evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- post-apply GT source for evaluation only: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`

## Method Changes Tested

### v184 Surface-Conditioned Residual U-Net

First real neural carrier. Inputs were parent RGB, normal, inverse depth, alpha, barycentric coordinates, texture, camera direction, and valid mask. It trained on train-fit Phase-J teacher residual and applied to target with stripped no-GT evidence.

Result: policy-val was only tiny-positive; exact did not pass.

### v185 GT-Anchored Surface U-Net

Same carrier, but train-fit GT losses were strengthened. This gave the first large policy-val all-axis gain and a real flowers exact improvement over the parent baked result.

Key command:

```bash
TMPDIR=/dev/shm/peilincai_tmp_v185 WANDB_MODE=offline WANDB_DIR=/dev/shm/peilincai_spcarnet_v185_gtanchored_unet/wandb CUDA_VISIBLE_DEVICES=2 PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_v185 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_conditioned_residual_unet.py --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented --output_dir /dev/shm/peilincai_spcarnet_v185_gtanchored_unet --steps 1600 --train_max_side 512 --patch_size 256 --base_channels 24 --lr 0.001 --teacher_l1_weight 0.25 --teacher_ssim_weight 0.05 --gt_l1_weight 1.0 --gt_ssim_weight 0.20 --delta_l1_weight 0.0002 --alpha_grid 0,0.125,0.25,0.375,0.5,0.75,1 --eval_tile 512 --eval_overlap 32 --ssim_max_side -1 --lpips_max_side 256 --compute_lpips --skip_policy_val_renders --method_name ours_26000_v185_gtanchored_unet_flowers --enable_wandb --wandb_project spcarnet_meshprior --wandb_run_name v185-gtanchored-unet-flowers-mid
```

### v186 Train-Fit Face Embedding U-Net

Added an optional train-only compact face-id embedding table. The LUT is collected only from train-fit face ids. Target/test faces not seen in train-fit map to zero embedding, so target evidence cannot create new learned capacity.

Verdict: negative. It increased scene-specific capacity but polluted policy-val structure. It failed the policy-val all-axis gate and target apply was skipped.

### v187/v188 Train-Only LPIPS Surface U-Net

Added differentiable train-only LPIPS losses to the v185 carrier. This directly addresses the v169 prompt's perceptual bottleneck rather than changing alpha or target footprint.

Implementation changes:

- `--gt_lpips_weight`
- `--teacher_lpips_weight`
- `--lpips_loss_max_side`
- optional `--face_embedding_dim`, disabled for v187/v188

v187 command used `--gt_lpips_weight 0.05`. v188 used `--gt_lpips_weight 0.10`.

v188 key command:

```bash
TMPDIR=/dev/shm/peilincai_tmp_v188 WANDB_MODE=offline WANDB_DIR=/dev/shm/peilincai_spcarnet_v188_lpips010_unet/wandb CUDA_VISIBLE_DEVICES=3 PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_v188 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_conditioned_residual_unet.py --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented --output_dir /dev/shm/peilincai_spcarnet_v188_lpips010_unet --steps 1600 --train_max_side 512 --patch_size 256 --base_channels 24 --lr 0.001 --max_delta 0.25 --teacher_l1_weight 0.20 --teacher_ssim_weight 0.05 --gt_l1_weight 0.9 --gt_ssim_weight 0.20 --gt_lpips_weight 0.10 --lpips_loss_max_side 128 --delta_l1_weight 0.0002 --alpha_grid 0,0.125,0.25,0.5,0.75,1,1.25,1.5 --eval_tile 512 --eval_overlap 32 --ssim_max_side -1 --lpips_max_side 256 --compute_lpips --skip_policy_val_renders --method_name ours_26000_v188_lpips010_unet_flowers --enable_wandb --wandb_project spcarnet_meshprior --wandb_run_name v188-lpips010-unet-flowers-mid
```

## Policy-Val Evidence

| run | policy-val verdict | alpha | PSNR gain | SSIM gain | LPIPS gain | positive-view fraction |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v185 gt-anchored U-Net | pass | 1.00 | +0.725499 | +0.049358 | +0.023701 | PSNR 1.0, SSIM 1.0, LPIPS 1.0 |
| v186 face embedding U-Net | fail | 0.50 | +0.012218 | -0.001373 | +0.008281 | PSNR 0.5, SSIM 0.4167, LPIPS 0.8333 |
| v187 LPIPS 0.05 | pass | 1.25 | +0.625793 | +0.051721 | +0.029317 | PSNR 1.0, SSIM 1.0, LPIPS 1.0 |
| v188 LPIPS 0.10 | pass | 1.25 | +0.721121 | +0.051595 | +0.033356 | PSNR 1.0, SSIM 1.0, LPIPS 1.0 |

Policy-val says LPIPS training is useful. It improves LPIPS gain over v185, but this still does not close flowers exact.

## Flowers Exact Results

Official evaluator:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/evaluate_render_split_metrics.py --model_path <run>/flowers_exact_target_apply --split test --methods <method_name> --output <run>/<version>_official_test_results.json --per_view_output <run>/<version>_official_test_per_view.json --merge_model_results
```

| run | PSNR | SSIM | LPIPS | vs Phase-J gate |
| --- | ---: | ---: | ---: | --- |
| parent/v168 exact reference | 19.832010 | 0.505779 | 0.405904 | fail all axes |
| v184 surface U-Net | 19.828789 | 0.505785 | 0.405885 | fail all axes |
| v185 gt-anchored U-Net | 20.225370 | 0.542942 | 0.368315 | fail all axes |
| v187 LPIPS 0.05 | 20.179890 | 0.545349 | 0.360239 | fail all axes |
| v188 LPIPS 0.10 | 20.214745 | 0.543694 | 0.361005 | fail all axes |
| Phase-J gate | 20.304358 | 0.557770 | 0.329222 | target |

Best exact by axis:

- PSNR: v185, `20.225370`, still `-0.078988` below Phase-J.
- SSIM: v187, `0.545349`, still `-0.012421` below Phase-J.
- LPIPS: v187, `0.360239`, still `+0.031017` worse than Phase-J.

## No-GT Audit

External verifier passed for v185, v187, and v188. Example forbidden keys:

- `rgb_gt`
- `residual_rgb`
- `teacher_residual_rgb`
- `teacher_residual_rgb_raw`
- `teacher_better_mask`
- `teacher_gain_l1`
- `teacher_parent_delta_l1`

GT was populated only after target apply:

- v187 audit: `/dev/shm/peilincai_spcarnet_v187_lpips005_unet/v187_eval_gt_population_audit.json`
- v188 audit: `/dev/shm/peilincai_spcarnet_v188_lpips010_unet/v188_eval_gt_population_audit.json`

## Interpretation

The new v169-guided route produced a real method improvement over v168/v184:

- v185 improved flowers exact from about `19.832 / 0.506 / 0.406` to `20.225 / 0.543 / 0.368`.
- v187/v188 improved the perceptual axis further, reaching best LPIPS `0.360239`.
- However, Phase-J remains out of reach. The remaining gap is not an alpha-only issue; earlier v185 posterior alpha probing improved LPIPS with larger alpha but sacrificed PSNR/SSIM and never reached the Phase-J gate.

The most important lesson is that policy-val overestimates target/test transfer. v187/v188 policy-val LPIPS gains are large and all-view positive, but target exact LPIPS improves only modestly. This points to a cross-view generalization bottleneck in the current carrier:

1. The U-Net learns a train-view perceptual correction.
2. The correction survives no-GT target apply enough to improve over v168.
3. The correction does not carry enough Phase-J high-frequency and view-dependent appearance to match the Phase-J endpoint.

## Verdict

Final status for this milestone: NOT COMPLETE.

The prompt's completion case A is not met because flowers exact does not beat Phase-J all-axis. Full9 must not be launched from this version.

Completion case B is partially supported: direct face memory fails policy-val, and LPIPS training improves but does not close exact. The current surface-conditioned U-Net carrier is useful but still under-capacity or under-informed for Phase-J-level perceptual transfer.

Recommended next experiment:

- Train a surface-attached low-rank feature texture with explicit view-conditioned coefficients, but keep v185/v187 as the fallback parent.
- The next carrier should preserve v185's PSNR while adding a target-support-aware perceptual branch. It should be certified on policy-val and then tested on flowers exact only.

