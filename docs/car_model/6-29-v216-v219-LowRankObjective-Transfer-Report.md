# v216-v219 Low-Rank Objective Transfer Report

Date: 2026-06-29

Prompt basis: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

Status: `NOT COMPLETE`. The new low-rank surface residual path made a real
method change and improved the local exact tradeoff, but it still fails the
v169 hard flowers gate because PSNR remains below Phase-J.

## Gate

Fixed Phase-J flowers gate:

| metric | required |
|---|---:|
| PSNR | `> 20.304358` |
| SSIM | `> 0.557770` |
| LPIPS | `< 0.329222` |

All exact runs below pass SSIM and LPIPS against this fixed gate, but all fail
PSNR. No full9 promotion is authorized.

## Implemented Method Changes

Files:

- `scripts/car_model/train_surface_lowrank_residual_texture.py`
- `scripts/car_model/apply_surface_lowrank_residual_texture.py`

New capabilities:

- target support / OOT confidence for low-rank surface residual transfer;
- train-fit slot reliability confidence, calibrated only from train-fit GT;
- gradient-weighted low-rank fitting using teacher-residual luma gradients;
- optional teacher residual target transform: `raw_rgb`, `luma_only`,
  `edge_luma_mix`;
- target apply audit fields for confidence, no-GT preflight, changed fraction,
  and effective confidence footprint.

The apply path still uses target evidence with RGB GT stripped. Target/test GT
is read only after candidate images are written, for metric evaluation.

## Policy-Val Evidence

| run | main change | alpha | PSNR gain | SSIM gain | LPIPS gain | pos frac PSNR/SSIM/LPIPS | full cosine | full retention |
|---|---|---:|---:|---:|---:|---|---:|---:|
| v217 | support + slot reliability | 2.0 | +0.034295 | +0.001206 | +0.000995 | 1.00 / 1.00 / 1.00 | 0.184937 | 0.068549 |
| v218 | gradient-weighted raw residual fit | 1.5 | +0.090935 | +0.003495 | +0.002624 | 1.00 / 1.00 / 1.00 | 0.282213 | 0.328257 |
| v219 | edge-luma residual target | 1.25 | +0.059447 | +0.002094 | +0.001345 | 1.00 / 1.00 / 1.00 | 0.240605 | 0.137082 |
| v220 | grid-8 gradient-weighted fit, face cap 131k | 2.0 | +0.063413 | +0.002703 | +0.002096 | 1.00 / 1.00 / 1.00 | 0.239664 | 0.242432 |

Policy-val conclusion:

- v217 made the policy safer but over-shrank the signal.
- v218 is the strongest policy-val result in this group and carries much more
  teacher residual energy.
- v219 confirms that edge/luma target transform is not better than weighted raw
  residual fitting under this low-rank carrier.
- v220 shows that simply increasing UV grid resolution is not enough when the
  face cap lowers global residual coverage. It improves active-bin retention but
  does not beat v218 policy-val, so no exact run was launched.

## Flowers Exact Evidence

| run | alpha | PSNR | SSIM | LPIPS | dPSNR vs parent | dSSIM | dLPIPS | changed frac | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v216 support | 1.5 | 19.884103 | 0.620712 | 0.179246 | +0.052049 | +0.000801 | +0.001089 | 0.309729 | FAIL |
| v216 support | 2.0 | 19.872846 | 0.619513 | 0.178797 | +0.040792 | -0.000397 | +0.001538 | 0.317497 | FAIL |
| v217 support + slot reliability | 2.0 | 19.854140 | 0.620257 | 0.179888 | +0.022086 | +0.000347 | +0.000447 | 0.095250 | FAIL |
| v218 gradient-weighted fit | 1.0 | 19.885165 | 0.621180 | 0.179533 | +0.053111 | +0.001269 | +0.000801 | 0.300343 | FAIL |
| v218 gradient-weighted fit | 1.5 | 19.881865 | 0.620270 | 0.178967 | +0.049811 | +0.000359 | +0.001367 | 0.314320 | FAIL |
| v219 edge-luma target | 1.25 | 19.871386 | 0.620453 | 0.179822 | +0.039332 | +0.000543 | +0.000513 | 0.171903 | FAIL |

Best exact interpretation:

- v218 alpha 1.0 is the best exact PSNR/SSIM point in this group and improves
  parent on all three metrics.
- v218 alpha 1.5 and v216 alpha 2.0 give better LPIPS, but lose PSNR/SSIM tradeoff.
- None reaches the required PSNR `20.304358`; the closest current exact result
  is still about `0.419` dB below the Phase-J PSNR gate.

## Artifact Index

Policy-val audits:

- v217: `/dev/shm/peilincai_spcarnet_v217_lowrank_support_slotrel_cov97/v212_lowrank_uv_residual_texture_audit.json`
- v218: `/dev/shm/peilincai_spcarnet_v218_lowrank_gradfit_support_cov97/v212_lowrank_uv_residual_texture_audit.json`
- v219: `/dev/shm/peilincai_spcarnet_v219_edge_luma_lowrank_support_smoke/v212_lowrank_uv_residual_texture_audit.json`

Exact audits:

- v217 alpha 2.0: `/dev/shm/peilincai_spcarnet_v217_lowrank_support_slotrel_cov97/flowers_exact_apply_alpha2/lowrank_target_apply_audit.json`
- v218 alpha 1.0: `/dev/shm/peilincai_spcarnet_v218_lowrank_gradfit_support_cov97/flowers_exact_apply_alpha1/lowrank_target_apply_audit.json`
- v218 alpha 1.5: `/dev/shm/peilincai_spcarnet_v218_lowrank_gradfit_support_cov97/flowers_exact_apply_alpha1p5/lowrank_target_apply_audit.json`
- v219 alpha 1.25: `/dev/shm/peilincai_spcarnet_v219_edge_luma_lowrank_support_smoke/flowers_exact_apply_alpha1p25/lowrank_target_apply_audit.json`
- v220 policy-only grid8 audit: `/dev/shm/peilincai_spcarnet_v220_grid8_gradfit_support_cov90/v212_lowrank_uv_residual_texture_audit.json`

W&B mode: all new medium/exact runs used `WANDB_MODE=offline`.

## Bottleneck

The bottleneck is no longer "teacher residual is absent" or "target apply leaks
GT". Teacher signal exists, no-GT apply is enforced, and the carrier can produce
positive policy-val gains. The remaining blocker is transfer strength: the
surface low-rank carrier still cannot lift flowers target PSNR to the Phase-J
gate while preserving exact SSIM/LPIPS.

Next required step:

- keep v218 gradient-aware fitting as the current low-rank baseline;
- do not rely on grid-size alone; v220 shows grid8 with lower face coverage is
  weaker than v218;
- test a representation that increases capacity without losing global residual
  coverage, for example a multiresolution residual residual-on-residual fit or a
  small surface feature decoder with a hard no-op support prior;
- do not launch full9 until flowers exact PSNR exceeds `20.304358` under the
  same no-GT protocol.
