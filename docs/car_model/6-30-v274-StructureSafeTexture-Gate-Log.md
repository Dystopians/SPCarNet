# v274 Structure-Safe Texture Low-Rank Gate Log

Date: 2026-06-30

## Purpose

This round follows `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

The intended method change was not another alpha scan or scalar confidence head.  v274 adds a structure-aware texture residual carrier:

- the source bank now stores `residual_edge`, `residual_luma_abs`, and `teacher_better_fraction`;
- the new decoder mode is `--residual_decoder_mode structure_safe_texture_lowrank`;
- the decoder fits the same face/UV-neighborhood low-rank teacher residual basis as the v270 texture branch, but gates texture injection by source/target edge agreement, residual-edge support, teacher-better support, and unique-source-view support;
- the expensive texture branch is prefiltered by the same structure gate before covariance/eigendecomposition, because naive full-resolution texture low-rank was too slow.

The hard gate remains unchanged:

- Phase-J flowers reference: `20.304358 / 0.557770 / 0.329222`
- Candidate must beat Phase-J all-axis before any full9 run.

## Code Changes

File changed:

- `scripts/car_model/train_surface_deferred_source_residual_renderer.py`

Implemented interfaces:

- checkpoint save/load for `residual_edge`, `residual_luma_abs`, `teacher_better_fraction`;
- backward-compatible fallback for old banks;
- new decoder choice `structure_safe_texture_lowrank`;
- policy/target support stats:
  - `mean_texture_structure_gate`
  - `p10_texture_structure_gate`
  - `mean_texture_edge_match`
  - `mean_texture_teacher_support`
- W&B offline flag for `residual_decoder/is_structure_safe_texture_lowrank`.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/train_surface_deferred_source_residual_renderer.py
git diff --check -- scripts/car_model/train_surface_deferred_source_residual_renderer.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_deferred_source_residual_renderer.py --help | rg -n "structure_safe_texture_lowrank"
```

All passed.

## Storage And GPU Preflight

Preflight before exact runs:

```text
/data:   28T size, 27T used, 119G available
/dev/shm: 252G size, 251G used, 1.5G available
/tmp:    root filesystem had about 6.0T available
```

GPU choice:

- used `CUDA_VISIBLE_DEVICES=5`;
- GPU 5 was low utilization at launch and W&B was set to offline mode via `WANDB_MODE=offline`;
- no new full evidence copy was made.

## Runs

### Interrupted Timing Runs

These runs are not quality evidence.

| run | reason |
|---|---|
| v274a | inherited full alpha grid and policy recalibration; interrupted at policy reliability `4/12` because it would be a multi-hour texture-lowrank scan rather than fixed-policy validation |
| v274b | fixed alpha but `eval_chunk_size=196608`; interrupted in first policy-val view after about 4.8 min due giant texture tensor construction |
| v274c | added structure prefilter but kept `eval_chunk_size=196608`; interrupted in first policy-val view after about 5.25 min, confirming large chunk concatenation was the bottleneck |

### Effective Runs

All effective runs used W&B offline and `target_eval_mode always`.

| run | bank / policy | method | policy-val PSNR / SSIM / LPIPS | target PSNR / SSIM / LPIPS | target gains | verdict |
|---|---|---|---:|---:|---:|---|
| v266c ref | v266c | `hybrid_edge_lowrank` | 20.668309 / 0.719789 / 0.152274 | 19.845698 / 0.620201 / 0.179915 | +0.013644 / +0.000290 / +0.000419 | reference |
| v270d ref | v270d | `hybrid_edge_texture_lowrank` | 20.673378 / 0.720244 / 0.152112 | 19.844320 / 0.620226 / 0.179934 | +0.012266 / +0.000315 / +0.000401 | texture reference |
| v274d | loaded v266c bank/policy | `structure_safe_texture_lowrank` | 20.668287 / 0.719788 / 0.152273 | 19.845704 / 0.620200 / 0.179917 | +0.013650 / +0.000290 / +0.000418 | tiny PSNR/tail improvement, not all-axis |
| v274e | fresh-fit bank + v274 stats | `structure_safe_texture_lowrank` | 20.675884 / 0.720313 / 0.151997 | 19.844540 / 0.620225 / 0.180015 | +0.012486 / +0.000314 / +0.000320 | policy-val win, target PSNR/LPIPS fail |
| v274f | loaded v270d bank/policy | `structure_safe_texture_lowrank` | 20.673402 / 0.720246 / 0.152113 | 19.844289 / 0.620224 / 0.179933 | +0.012235 / +0.000314 / +0.000402 | matches v270 tradeoff, not better |

Target exact output roots:

- v274d: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v274_structure_safe_texture_flowers_20260630/v274d_v266policy_structure_prefilter_chunk65536_flowers`
- v274e: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v274_structure_safe_texture_flowers_20260630/v274e_freshfit_structure_stats_fixedalpha_flowers`
- v274f: `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v274_structure_safe_texture_flowers_20260630/v274f_v270policy_structure_prefilter_chunk65536_flowers`

All three no-target-GT audits passed:

```text
uses_target_or_test_gt: false
uses_target_or_test_gt_after_apply_for_eval: true
```

## Phase-J Gate

None passed Phase-J flowers all-axis.

| run | PSNR minus Phase-J | SSIM minus Phase-J | Phase-J LPIPS minus candidate | gate |
|---|---:|---:|---:|---|
| v274d | -0.458654 | +0.062430 | +0.149305 | fail PSNR |
| v274e | -0.459818 | +0.062455 | +0.149207 | fail PSNR |
| v274f | -0.460069 | +0.062454 | +0.149289 | fail PSNR |

## Interpretation

v274 is a valid representation-level implementation, but it is not a breakthrough.

What improved:

- The new texture carrier can be run under the strict no-target-GT protocol.
- Structure prefiltering fixed the worst runtime behavior of naive texture low-rank at large chunks.
- v274d slightly improved target PSNR over v266c by `+0.000006` and PSNR tail CVaR by about `+0.000013`, so the gate does not catastrophically damage the strongest PSNR reference.

What failed:

- The improvement is far below meaningful paper-level effect size.
- v274d loses tiny amounts of SSIM/LPIPS against v266c.
- v274e proves that fresh residual-edge/teacher-better statistics can overfit policy-val: policy-val becomes much stronger, but target PSNR/LPIPS get worse.
- v274f does not improve the v270 texture tradeoff.
- The method remains about `0.459` PSNR below the Phase-J flowers reference.

## Current Bottleneck

The current source/UV-bin texture family is too local.  Even when edge-safe and teacher-supported, it mostly redistributes the same weak source-slot residuals.  The missing signal appears to be a more global, view-dependent, high-frequency appearance field, not a safer local blend of existing residual slots.

Do not promote v274 to full9.

Recommended next direction:

- stop adding scalar gates to the current source-slot carrier;
- train a compact view-dependent surface feature decoder whose features are learned from teacher-parent residual patches, not only aggregated source-slot RGB means;
- add a projection diagnostic that reports teacher residual energy/cosine separately for low-frequency and high-frequency bands;
- only return to full flowers exact after that carrier beats v266c/v270d by a meaningful margin on target exact, not just policy-val.

Final status: `NOT COMPLETE`.
