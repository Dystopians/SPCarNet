# v271 Source-View Consistency Gate Log

Date: 2026-06-30

## Verdict

Final status: **NOT COMPLETE**.

v271 implements the next diagnostic mechanism after v270: source-view
leave-one-out residual consistency. It is a real train/eval pipeline method
change, but it does not beat the previous flowers best and does not pass the
Phase-J flowers PSNR gate.

Best previous baked flowers result:

```text
v266c: 19.845698 PSNR / 0.620201 SSIM / 0.179915 LPIPS
```

Best v271 tradeoff:

```text
v271c: 19.845337 PSNR / 0.620191 SSIM / 0.179887 LPIPS
```

v271c improves LPIPS over v266c, but loses PSNR and SSIM. v271d nearly restores
PSNR/SSIM, but loses the LPIPS improvement. Therefore v271 is not a new
all-axis best.

## Method

Implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

New CLI:

```text
--source_consistency_mode {off,weight,weight_amplitude}
--source_consistency_min_other_sources
--source_consistency_error_beta
--source_consistency_floor
--source_consistency_amplitude_floor
--source_consistency_amplitude_max
```

The mechanism:

1. For each train-fit source residual slot in a face/UV bin, hold that source
   view out.
2. Predict its residual from the other source-view slots using view, normal,
   parent RGB, support count, and teacher-gain weights.
3. Convert leave-one-out cosine and relative error into a source-slot
   reliability value.
4. Freeze the reliability map before policy-val/target apply.
5. Use stripped target no-GT evidence for application; load target GT only after
   apply for exact metrics.

This tests whether source-view consistency can distinguish reliable teacher
residual directions from target-trajectory-unsafe residuals.

## Storage And Runtime

Preflight from the run:

```text
/data    avail about 125G
/dev/shm avail about 1.4G
/tmp     avail about 6.0T
```

Because `/dev/shm` remained nearly full, all runs read the existing low-copy
evidence cache and wrote outputs under `/data`. W&B was enabled in offline mode.

## Commands

v271b, source consistency on v270d-style `hybrid_edge_texture_lowrank`:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/peilincai/mesh-splatting/outputs/tmp PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_deferred_source_residual_renderer.py \
  --bank_checkpoint /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --policy_val_stride 4 --grid 4 --min_source_count 2 \
  --source_consistency_mode weight --source_consistency_error_beta 0.75 --source_consistency_floor 0.35 \
  --residual_decoder_mode hybrid_edge_texture_lowrank \
  --local_linear_l2 0.05 --local_linear_blend 1.0 --local_linear_min_sources 3 \
  --lowrank_basis_rank 3 --lowrank_basis_min_sources 4 --lowrank_basis_min_unique_views 3 \
  --lowrank_basis_l2 0.05 --lowrank_basis_blend 0.15 --lowrank_basis_disagreement_beta 4.0 \
  --patch_coherent_radius 1 --patch_coherent_bin_sigma 0.9 \
  --alpha_grid 0,0.00390625,0.0078125,0.015625,0.03125,0.046875,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1.0 \
  --eval_chunk_size 8192 --compute_lpips --target_eval_mode auto --enable_wandb \
  --output_dir /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v271_source_consistency_flowers_20260630/v271b_weight_fullflowers
```

v271c/v271d, source consistency on the previous best v266c base:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/peilincai/mesh-splatting/outputs/tmp PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_deferred_source_residual_renderer.py \
  --bank_checkpoint /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --policy_val_stride 4 --grid 4 --min_source_count 2 \
  --source_consistency_mode weight --source_consistency_error_beta 0.75 --source_consistency_floor 0.35 \
  --residual_decoder_mode hybrid_edge_lowrank \
  --local_linear_l2 0.05 --local_linear_blend 1.0 --local_linear_min_sources 3 \
  --lowrank_basis_rank 3 --lowrank_basis_min_sources 4 --lowrank_basis_min_unique_views 3 \
  --lowrank_basis_l2 0.05 --lowrank_basis_blend 0.20 --lowrank_basis_disagreement_beta 8.0 \
  --policy_reliability_mode patch_perceptual_v1 --policy_reliability_alpha 0.03125 \
  --policy_reliability_min_count 8 --policy_reliability_min_positive_fraction 0.48 \
  --policy_gain_mode positive_soft --policy_gain_max 2.0 --policy_gain_scale 0.000025 \
  --alpha_grid 0,0.00390625,0.0078125,0.015625,0.03125,0.046875,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1.0 \
  --eval_chunk_size 65536 --compute_lpips --target_eval_mode auto --enable_wandb
```

For v271d, the only difference is:

```text
--source_consistency_floor 0.70
```

## Results

| run | base | consistency floor | exact PSNR | exact SSIM | exact LPIPS | PSNR gain | SSIM gain | LPIPS gain | Phase-J PSNR gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v266c | hybrid_edge_lowrank | n/a | 19.845698 | 0.620201 | 0.179915 | +0.013644 | +0.000290 | +0.000419 | -0.458660 |
| v270d | hybrid_edge_texture_lowrank | n/a | 19.844320 | 0.620226 | 0.179934 | +0.012266 | +0.000315 | +0.000401 | -0.460038 |
| v271b | v270d + consistency | 0.35 | 19.844153 | 0.620207 | 0.179937 | +0.012099 | +0.000296 | +0.000398 | -0.460205 |
| v271c | v266c + consistency | 0.35 | 19.845337 | 0.620191 | 0.179887 | +0.013283 | +0.000281 | +0.000448 | -0.459021 |
| v271d | v266c + mild consistency | 0.70 | 19.845648 | 0.620200 | 0.179919 | +0.013594 | +0.000290 | +0.000416 | -0.458710 |

## Interpretation

The LOO signal is meaningful but not sufficient as a direct multiplicative
source weight:

- v271c improves LPIPS over v266c, showing that source-view consistency does
  capture some perceptual-risk information.
- The same weighting reduces PSNR and SSIM, so it removes useful teacher signal.
- Mild weighting in v271d nearly recovers v266c PSNR/SSIM, but the LPIPS
  improvement disappears.
- No v271 variant is a new all-axis best, and all remain far below Phase-J PSNR.

The next method should not use LOO consistency as a hard source-slot weight.
It should use it as a feature in a learned confidence/amplitude head trained on
policy-val outcomes, or as an uncertainty prior that affects only high-risk
regions.

## Artifacts

```text
docs/car_model/results/v271_source_view_consistency_summary.json
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v271_source_consistency_flowers_20260630/v271b_weight_fullflowers/v253_deferred_source_renderer_audit.json
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v271_source_consistency_flowers_20260630/v271c_weight_v266base_fullflowers/v253_deferred_source_renderer_audit.json
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v271_source_consistency_flowers_20260630/v271d_weight070_v266base_fullflowers/v253_deferred_source_renderer_audit.json
```

## Next Step

Use `source_consistency_reliability`, LOO cosine/error, policy reliability,
tail risk, parent mismatch, and support features as inputs to a learned
policy-val confidence/amplitude head. Do not directly multiply source weights
by LOO consistency unless a held-out target-style certificate proves it helps
all-axis.
