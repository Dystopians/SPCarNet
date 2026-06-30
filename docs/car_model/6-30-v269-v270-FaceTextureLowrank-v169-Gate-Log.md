# v269-v270 Face-Texture Low-Rank Distillation Gate Log

Date: 2026-06-30

Prompt followed:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

## Verdict

Final status: **NOT COMPLETE**.

The v169-inspired representation upgrade was implemented and verified on
flowers policy-val/exact, but it does **not** pass the hard v169 flowers gate.
The best run in this batch, `v270d_hybrid_edge_texture_fullalpha_flowers`, is
positive vs parent on flowers exact, but remains below Phase-J on PSNR:

```text
v270d exact: 19.844320 PSNR / 0.620226 SSIM / 0.179934 LPIPS
Phase-J ref: 20.304358 PSNR / 0.557770 SSIM / 0.329222 LPIPS
gap:         -0.460038 PSNR / +0.062456 SSIM / +0.149288 LPIPS-direction
```

No full9 was launched because the prompt explicitly blocks full9 before
flowers exact beats Phase-J all-axis.

## What Changed

Implementation file:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

New/finished decoder modes:

```text
patch_coherent_hybrid
face_texture_lowrank
hybrid_edge_texture_lowrank
```

The important v169-oriented change is `face_texture_lowrank`:

- gather train-fit Phase-J teacher residual samples from neighboring UV bins on
  the same mesh face;
- weight them by UV distance, view direction, normal agreement, parent RGB,
  parent edge, support count, and teacher gain;
- fit a compact RGB low-rank basis per target batch from the coherent same-face
  samples;
- predict target coefficients from view, parent appearance, edge, and relative
  UV offset features;
- apply the residual only to stripped target no-GT evidence, then load target GT
  after apply for final exact evaluation.

`hybrid_edge_texture_lowrank` keeps the previous stable edge-local-linear
decoder as the base and injects the coherent face-texture low-rank carrier as a
controlled residual correction. This was needed because pure face texture was
too diluted on target.

## Storage And Runtime Preflight

```text
/data    avail 129G
/dev/shm avail 1.6G
/tmp     avail 6.0T
```

Decision:

- do not duplicate evidence under `/dev/shm`;
- read the existing low-copy evidence tree;
- write new outputs under `/data/peilincai/mesh-splatting/outputs`;
- run with `WANDB_MODE=offline`;
- use GPU 1, which was low-memory and low-utilization at launch.

## Commands

All commands used the existing loaded bank:

```text
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz
```

v269a was intentionally aborted after the first calibration view did not finish
after about 80 seconds:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/peilincai/mesh-splatting/outputs/tmp PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_deferred_source_residual_renderer.py \
  --residual_decoder_mode face_texture_lowrank \
  --lowrank_basis_blend 0.5 --patch_coherent_radius 2 --patch_coherent_bin_sigma 1.25 \
  --policy_reliability_mode patch_perceptual_v1 --target_eval_mode auto --enable_wandb \
  --output_dir /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v269_face_texture_flowers_20260630/v269a_face_texture_lowrank_loadedbank
```

v269c was the first complete pure face-texture exact run:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/peilincai/mesh-splatting/outputs/tmp PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_deferred_source_residual_renderer.py \
  --bank_checkpoint /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --policy_val_stride 4 --grid 4 --min_source_count 2 \
  --residual_decoder_mode face_texture_lowrank \
  --lowrank_basis_rank 3 --lowrank_basis_min_sources 4 --lowrank_basis_min_unique_views 3 \
  --lowrank_basis_l2 0.05 --lowrank_basis_blend 0.35 --lowrank_basis_residual_clip 0.12 \
  --lowrank_basis_disagreement_beta 2.0 --patch_coherent_radius 1 --patch_coherent_bin_sigma 0.9 \
  --policy_reliability_mode off --policy_gain_mode off --ood_gain_mode off \
  --alpha_grid 0,0.015625,0.03125,0.0625,0.125 \
  --eval_chunk_size 8192 --compute_lpips --target_eval_mode auto --enable_wandb \
  --output_dir /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v269_face_texture_flowers_20260630/v269c_face_texture_lowrank_fullflowers
```

v270d is the fairest hybrid run in this batch because it restores the full
alpha grid up to `1.0`, matching the previous v266c selection scope:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/peilincai/mesh-splatting/outputs/tmp PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_deferred_source_residual_renderer.py \
  --bank_checkpoint /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --policy_val_stride 4 --grid 4 --min_source_count 2 \
  --residual_decoder_mode hybrid_edge_texture_lowrank \
  --local_linear_l2 0.05 --local_linear_blend 1.0 --local_linear_min_sources 3 \
  --lowrank_basis_rank 3 --lowrank_basis_min_sources 4 --lowrank_basis_min_unique_views 3 \
  --lowrank_basis_l2 0.05 --lowrank_basis_blend 0.15 --lowrank_basis_residual_clip 0.12 \
  --lowrank_basis_disagreement_beta 4.0 --patch_coherent_radius 1 --patch_coherent_bin_sigma 0.9 \
  --policy_reliability_mode off --policy_gain_mode off --ood_gain_mode off \
  --alpha_grid 0,0.00390625,0.0078125,0.015625,0.03125,0.046875,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1.0 \
  --eval_chunk_size 8192 --compute_lpips --target_eval_mode auto --enable_wandb \
  --output_dir /data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v270_hybrid_edge_texture_flowers_20260630/v270d_hybrid_edge_texture_fullalpha_flowers
```

## Quantitative Results

| run | mode | alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | exact PSNR | exact SSIM | exact LPIPS | exact PSNR gain | exact SSIM gain | exact LPIPS gain | Phase-J PSNR gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v266c | hybrid_edge_lowrank | 1.000 | +0.061872 | +0.002263 | +0.001043 | 19.845698 | 0.620201 | 0.179915 | +0.013644 | +0.000290 | +0.000419 | -0.458660 |
| v269c | face_texture_lowrank | 0.125 | +0.009749 | +0.000394 | +0.000137 | 19.834773 | 0.620011 | 0.180294 | +0.002719 | +0.000101 | +0.000041 | -0.469585 |
| v270c | hybrid_edge_texture_lowrank | 0.250 | +0.022109 | +0.000896 | +0.000320 | 19.837958 | 0.620131 | 0.180231 | +0.005904 | +0.000220 | +0.000104 | -0.466400 |
| v270d | hybrid_edge_texture_lowrank | 1.000 | +0.066941 | +0.002718 | +0.001205 | 19.844320 | 0.620226 | 0.179934 | +0.012266 | +0.000315 | +0.000401 | -0.460038 |

v270d improves policy-val over v266c and has a tiny SSIM edge on exact, but it
loses exact PSNR and LPIPS to v266c. Therefore it is not a new overall best.

## Projection Diagnostics

| run | selected alpha | full energy retention | full cosine | active energy retention | active cosine | texture active/support |
|---|---:|---:|---:|---:|---:|---:|
| v269c | 0.125 | 0.005681 | 0.296857 | 0.005996 | 0.304822 | 0.558111 |
| v270d | 1.000 | 0.455350 | 0.316950 | 0.480498 | 0.325421 | 0.558111 |
| v266c | 1.000 | 0.515831 | 0.308118 | n/a | n/a | 0.000000 |

Interpretation:

- pure face texture was too conservative after policy selection;
- the hybrid version restores residual energy, but its target PSNR is still
  weaker than the previous edge-lowrank carrier;
- the coherent UV texture basis improves structural/policy-val behavior but
  does not solve the Phase-J PSNR gap.

## No-GT Audit

The target no-GT verifier passed in the completed exact runs. Forbidden keys
such as `rgb_gt`, `teacher_residual_rgb`, `teacher_gain_l1`, and related target
GT/residual fields were absent from:

```text
/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt
```

Target GT was loaded only after no-GT apply for exact evaluation.

## Artifacts

```text
docs/car_model/results/v269_v270_face_texture_lowrank_summary.json
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v269_face_texture_flowers_20260630/v269c_face_texture_lowrank_fullflowers/v253_deferred_source_renderer_audit.json
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v269_face_texture_flowers_20260630/v269c_face_texture_lowrank_fullflowers/target_exact_fixed_policy
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v270_hybrid_edge_texture_flowers_20260630/v270d_hybrid_edge_texture_fullalpha_flowers/v253_deferred_source_renderer_audit.json
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v270_hybrid_edge_texture_flowers_20260630/v270d_hybrid_edge_texture_fullalpha_flowers/target_exact_fixed_policy
/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v270_hybrid_edge_texture_flowers_20260630/v270d_hybrid_edge_texture_fullalpha_flowers/wandb/offline-run-20260629_233947-arf2zxuk
```

## Lessons

1. The v169 prompt was correct to demand a flowers gate before full9. This run
   shows how easy it is to get policy-val all-axis success without target PSNR
   closing the Phase-J gap.
2. Same-face low-rank texture capacity is useful but not decisive. It improves
   policy-val and exact SSIM, but target PSNR still prefers the older
   edge-lowrank carrier.
3. The next real method should not be another alpha or local UV-neighborhood
   variant. It needs a stronger teacher-student objective, likely with explicit
   target-trajectory robustness: held-out source-view validation, uncertainty
   over teacher residual directions, or a compact learned decoder trained to
   predict residual confidence and amplitude jointly rather than projecting
   residual color alone.
4. Full9 remains blocked. The only valid next promotion path is a new flowers
   exact result with `PSNR > 20.304358`, `SSIM > 0.557770`, and
   `LPIPS < 0.329222` under the same no-target-GT protocol.
