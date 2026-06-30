# v289 Target-Compatible Source Aggregation Gate Log

Date: 2026-06-30

Prompt reference: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

## Verdict

`NOT COMPLETE`.

v289 implements a real train/eval pipeline method change in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`: target-compatible source aggregation for the deferred source residual renderer. It improves the v286b recalibrated deferred-source baseline slightly on flowers exact, but it still fails the v169 Phase-J flowers gate because PSNR remains far below `20.304358`.

No full9 run is allowed from this branch.

## Storage And Runtime Preflight

- `/data`: about `108G` available before the run.
- `/dev/shm`: about `1.7G` available before the run.
- `/tmp` filesystem: about `6.0T` available before the run.
- The run reads existing low-copy evidence from `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers` and writes only run artifacts to `/data`.
- W&B mode: offline.
- GPUs: v289a and v289c used GPU1, v289b used GPU2.

## Method Change

The previous `view_feature_ridge_texture` decoder mixed train-fit Phase-J teacher residual source slots using view, normal, parent RGB, support count, source gain, self-error, and source-heldout confidence. v289 adds an explicit target compatibility layer before source aggregation:

- target-view source reweighting: downweights source slots whose camera direction is less compatible with the target view;
- optional target-view minimum cosine: soft or hard suppression for out-of-support source views;
- optional row confidence shrink: estimates row risk from view gap, parent RGB mismatch, edge mismatch, residual disagreement, effective source count, and unique source-view count;
- target no-GT apply remains strict: only `rgb_render`, geometry, normal, alpha, barycentric, and camera center are read from target apply evidence.

The new public CLI includes:

- `--target_compatibility_mode {off,soft,hard}`
- `--target_compatibility_view_sharpness`
- `--target_compatibility_min_view_cos`
- `--target_compatibility_beta`
- `--target_compatibility_floor`
- `--target_compatibility_min_effective_sources`
- `--target_compatibility_{view,parent,edge,variance,effective,unique_view}_weight`

## Commands And Artifacts

Exact commands are saved in each audit Markdown:

- v289a audit: `outputs/carnet/spcarnet_v289_targetcompat_20260630/v289a_soft_targetcompat_flowers/v253_deferred_source_renderer_audit.md`
- v289b audit: `outputs/carnet/spcarnet_v289_targetcompat_20260630/v289b_stronger_targetcompat_flowers/v253_deferred_source_renderer_audit.md`
- v289c audit: `outputs/carnet/spcarnet_v289_targetcompat_20260630/v289c_weightonly_targetcompat_flowers/v253_deferred_source_renderer_audit.md`

Shared evidence/config:

- bank checkpoint: `outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz`
- fit evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- target no-GT evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- target eval evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`
- fixed alpha grid: `1.0`
- target eval mode: `auto`
- LPIPS: enabled
- W&B project: `spcarnet-v169-target-compatibility`

## Results

Reference:

- Phase-J flowers gate: `20.304358 / 0.557770 / 0.329222`.
- v286b recalibrated deferred-source exact: `19.840910 / 0.620183 / 0.180100`.

| run | target compatibility | policy-val candidate | exact candidate | exact gain vs parent | PSNR gap vs Phase-J | no-GT audit | verdict |
|---|---|---:|---:|---:|---:|---|---|
| v286b | off | 20.650730 / 0.719454 / 0.152572 | 19.840910 / 0.620183 / 0.180100 | +0.008856 / +0.000272 / +0.000235 | -0.463448 | pass | reference fail |
| v289a | soft weighting + mild confidence shrink | 20.652035 / 0.719490 / 0.152559 | 19.841450 / 0.620205 / 0.180094 | +0.009396 / +0.000294 / +0.000241 | -0.462908 | pass | fail |
| v289b | sharper weighting + stronger confidence shrink | 20.650309 / 0.719392 / 0.152597 | 19.841702 / 0.620217 / 0.180109 | +0.009648 / +0.000306 / +0.000226 | -0.462656 | pass | fail |
| v289c | source weighting only, no compatibility shrink | 20.654506 / 0.719583 / 0.152522 | 19.841839 / 0.620214 / 0.180080 | +0.009785 / +0.000303 / +0.000255 | -0.462519 | pass | fail |

Best v289 result is v289c:

- exact PSNR improves over v286b by `+0.000929`;
- exact SSIM improves over v286b by `+0.000031`;
- exact LPIPS improves over v286b by `+0.000020`;
- exact PSNR is still `-0.462519` below Phase-J flowers.

## Diagnostic Interpretation

v289 answers a narrow but useful question:

> Is target-compatible source aggregation the missing mechanism that closes the Phase-J gap?

Answer: no.

The source weighting itself is a positive contribution: v289c beats v286b and v289a/b on policy-val and exact. The confidence shrink branch is not the right main mechanism here; it reduces effective residual energy and slightly weakens the best exact result. However, even the best source-weight-only result changes flowers exact by less than `0.001 dB` PSNR over v286b. This is not a scale problem that can plausibly be fixed by another local weight or confidence scan.

The persistent bottleneck remains the same:

- the active target changed fraction is only about `0.033`;
- the baked/deferred carrier still transfers too little correct Phase-J RGB residual energy;
- target-compatible aggregation can pick sources more sensibly, but it does not create stronger high-frequency teacher residual capacity.

## v169 Gate Status

- policy-val all-axis: pass for v289a, v289b, v289c.
- target no-GT audit: pass for v289a, v289b, v289c.
- flowers exact vs parent: pass for v289a, v289b, v289c.
- flowers exact vs Phase-J all-axis: fail for v289a, v289b, v289c due to PSNR.
- full9 allowed: no.

## Next Recommendation

Keep the source-weighting portion of v289 as a small positive component, but do not continue scanning compatibility confidence strength. The next real attempt must increase representation capacity or residual supervision strength, for example a patch-aware learned view-dependent surface decoder that predicts a higher-energy residual while using source-heldout direction checks as a certificate rather than a scalar shrink.

