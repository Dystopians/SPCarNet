# v169 Surface Decoder Gate Report

Date: 2026-06-29

This report records the v169 follow-up to `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

## Decision

Do not launch flowers exact or full9 from this branch yet.

The new v205-v208 route made a real representation-level change and improved the policy-val proxy, but it still fails the v169 source-projection gate. The best current run is v208. It improves policy-val all-axis at alpha 0.25, but its Phase-J teacher projection is still weak:

| run | policy-val PSNR gain | policy-val SSIM gain | policy-val LPIPS gain | full cosine | full energy retention | active cosine | active energy retention | projection SSIM vs teacher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v204 prior best baked decoder | +0.012003 | +0.000090 | +0.000229 | 0.089507 | 0.021860 | n/a | n/a | -0.000106 |
| v205 Fourier direction | +0.012107 | +0.000064 | +0.000250 | 0.083013 | 0.025752 | 0.150089 | 0.085368 | -0.000245 |
| v206 Fourier image proxy | +0.012631 | +0.000062 | +0.000254 | 0.082032 | 0.030105 | 0.147234 | 0.099496 | -0.000281 |
| v207 Fourier energy coverage | +0.014938 | +0.000049 | +0.000137 | 0.094705 | 0.026670 | 0.149643 | 0.066415 | -0.000129 |
| v208 high coverage Fourier | +0.017802 | +0.000118 | +0.000210 | 0.100149 | 0.024963 | 0.138292 | 0.047590 | -0.000100 |

The intended v169 source-projection target remains roughly cosine >= 0.25, energy retention >= 0.25, and no SSIM/LPIPS regression against the Phase-J teacher. v208 is the best proxy and best full cosine row, but it is still far from that gate.

## Implemented Method Changes

Files changed:

- `scripts/car_model/train_perceptual_surface_residual_decoder.py`
- `scripts/car_model/audit_surface_checkpoint_residual_projection.py`

Implemented additions:

- `feature_mode=fourier_v1`: Fourier barycentric/local coordinates plus normal-camera interaction features.
- Residual-aware training objective: weighted RGB/luma loss, optional image proxy, cosine direction loss, and energy matching loss.
- Residual-energy face policy: `--candidate_target_energy_coverage` selects faces by cumulative `teacher_residual_rgb` energy instead of only a fixed count.
- Projection audit support for `--checkpoint_type perceptual_surface_decoder`.
- Projection audit now reports both full valid surface residual stats and selected-active residual stats. This separates carrier coverage failure from decoder fit failure.

## Experiments

All runs used W&B offline logging and `/dev/shm` outputs.

| run | output dir | W&B offline dir | main setting |
|---|---|---|---|
| v205 | `/dev/shm/peilincai_spcarnet_v205_fourier_direction_decoder` | `/dev/shm/peilincai_spcarnet_v205_fourier_direction_decoder/wandb/offline-run-20260629_044217-dk276qgy` | Fourier features, 32k faces, direction/energy losses |
| v206 | `/dev/shm/peilincai_spcarnet_v206_fourier_image_decoder` | `/dev/shm/peilincai_spcarnet_v206_fourier_image_decoder/wandb/offline-run-20260629_044217-otbl77dr` | v205 plus image proxy |
| v207 | `/dev/shm/peilincai_spcarnet_v207_fourier_energycoverage_decoder` | `/dev/shm/peilincai_spcarnet_v207_fourier_energycoverage_decoder/wandb/offline-run-20260629_044748-jl9vyy36` | 65k energy-coverage faces |
| v208 | `/dev/shm/peilincai_spcarnet_v208_highcoverage_fourier_decoder` | `/dev/shm/peilincai_spcarnet_v208_highcoverage_fourier_decoder/wandb/offline-run-20260629_045640-qoa3cgys` | 131k high-coverage faces |

Important artifacts:

- Summary JSON: `docs/car_model/results/v169_v200_v208_projection_summary.json`
- v208 audit: `/dev/shm/peilincai_spcarnet_v208_highcoverage_fourier_decoder/v180_perceptual_surface_decoder_audit.json`
- v208 projection audit: `/dev/shm/peilincai_spcarnet_v208_highcoverage_fourier_decoder/v208_projection_audit_alpha025.json`

## Key Commands

Representative v208 training command:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/dev/shm/peilincai_tmp_v208_fourier WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --policy_val_stride 4 --residual_rgb_key teacher_residual_rgb --residual_l1_key teacher_residual_l1 \
  --min_l1 0.0005 --min_alpha 0.03 --max_candidate_faces 131072 --candidate_target_energy_coverage 0.90 \
  --max_candidate_face_samples_per_view 65536 --batch_size 65536 --steps 450 --lr 0.0008 \
  --embedding_dim 24 --hidden_dim 192 --layers 4 --max_delta 0.20 --feature_mode fourier_v1 \
  --image_loss_every 0 --sample_weight_gamma 0.5 --sample_weight_clip 8 \
  --cosine_loss_weight 0.08 --energy_match_weight 0.80 --mag_reg 0.0 \
  --compute_lpips --output_dir /dev/shm/peilincai_spcarnet_v208_highcoverage_fourier_decoder \
  --enable_wandb --wandb_run_name v208-highcoverage-fourier-decoder --seed 208
```

Representative v208 projection audit:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/dev/shm/peilincai_tmp_v208_projection025 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/audit_surface_checkpoint_residual_projection.py \
  --run_name v208_highcoverage_fourier_decoder_alpha025 --checkpoint_type perceptual_surface_decoder \
  --checkpoint /dev/shm/peilincai_spcarnet_v208_highcoverage_fourier_decoder/v180_perceptual_surface_decoder.pt \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --residual_rgb_key teacher_residual_rgb --residual_l1_key teacher_residual_l1 \
  --alpha 0.25 --policy_val_stride 4 --min_alpha 0.03 --min_l1 0.0005 \
  --compute_lpips --output_json /dev/shm/peilincai_spcarnet_v208_highcoverage_fourier_decoder/v208_projection_audit_alpha025.json \
  --output_md /dev/shm/peilincai_spcarnet_v208_highcoverage_fourier_decoder/v208_projection_audit_alpha025.md
```

## Lessons

The main bottleneck is no longer just protocol or alpha selection. Even selected-active pixels only reach about 0.14-0.15 cosine and <=0.10 energy retention. Increasing selected faces from 32k to 131k improves proxy PSNR and full cosine, but it does not solve residual direction fidelity.

The next method should stop expanding this smooth face-embedding MLP. The more promising v169-compliant path is an explicit local surface carrier: per-face UV micro-texture, low-rank residual basis, or a tiled residual field with view-conditioned coefficients. That would directly increase local residual capacity instead of asking one global MLP to memorize sparse high-frequency teacher corrections.
