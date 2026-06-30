# v272 Learned Source-Consistency Head Gate Log

Date: 2026-06-30

Prompt followed: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

## Purpose

v272 tests whether source-view consistency can be used as a learned, target-free decision signal rather than as another hand-tuned hard gate. The motivation is that v271 hard-multiplied source consistency into weights and slightly traded PSNR/SSIM for LPIPS. v272 therefore keeps the consistency map as a feature for a policy-val-supervised head and checks whether the learned policy generalizes to flowers target exact.

This is a real train/eval pipeline change, but it is not promoted as the main paper method because target exact did not improve over the best previous flowers target runs.

## Code Changes

Edited file:

- `scripts/car_model/train_surface_deferred_source_residual_renderer.py`

Implemented:

- `--source_consistency_mode feature_only`.
- Checkpoint save/load for `source_consistency_apply_weight` and `source_consistency_apply_amplitude`.
- Backward-compatible hard-apply behavior for older checkpoints that contain consistency maps but no apply flags.
- Learned OOD/gain features now include:
  - source consistency reliability,
  - source consistency amplitude,
  - source consistency gap,
  - base confidence,
  - raw residual magnitude.
- `--learned_ood_head_ceiling` so the learned head can be used as shrink-only, mild, or boost-only instead of always clipping to `<= 1`.
- Markdown/W&B logging for learned head floor and ceiling.

Validation:

- `PYTHONDONTWRITEBYTECODE=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/train_surface_deferred_source_residual_renderer.py`
- `git diff --check -- scripts/car_model/train_surface_deferred_source_residual_renderer.py`
- CLI help confirmed `feature_only` and `learned_ood_head_ceiling` are exposed.

## Storage / Runtime Preflight

Snapshot before exact experiments:

- `/data`: 28T total, 27T used, 121G available.
- `/dev/shm`: 252G total, 251G used, 1.5G available.
- `/tmp`: 14T total, 7.2T used, 6.0T available.

Because `/dev/shm` was nearly full, all new outputs were written to:

- `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v272_learned_consistency_flowers_20260630`

Existing low-copy evidence under `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers` was read but not duplicated. W&B was run in offline mode for all medium/full runs.

## Experiment Matrix

All exact command lines are preserved in each run's audit JSON under the `command` field.

Common evidence paths:

- fit: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- target no-GT: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- target eval: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`

| run | base bank | change | policy PSNR / SSIM / LPIPS | target PSNR / SSIM / LPIPS | target gains | verdict |
|---|---|---|---:|---:|---:|---|
| v266c | v266c | previous best conservative hybrid | 20.668309 / 0.719789 / 0.152274 | 19.845698 / 0.620201 / 0.179915 | +0.013644 / +0.000290 / +0.000419 | reference |
| v270d | v270d | previous texture-lowrank front | 20.673378 / 0.720244 / 0.152112 | 19.844320 / 0.620226 / 0.179934 | +0.012266 / +0.000315 / +0.000401 | reference |
| v272a | v266c | feature-only consistency + learned head smoke, stride 8, no target exact | 20.901913 / 0.747219 / 0.139397 | n/a | policy +0.070305 / +0.002550 / +0.001575 | smoke pass |
| v272b | v266c | learned head floor 0.65, ceiling 1.05 | 20.672710 / 0.720170 / 0.152122 | 19.843843 / 0.620191 / 0.179945 | +0.011789 / +0.000281 / +0.000390 | target fail |
| v272c | v266c | milder learned head floor 0.85, ceiling 1.03 | 20.673808 / 0.720213 / 0.152094 | 19.844036 / 0.620193 / 0.179934 | +0.011983 / +0.000282 / +0.000401 | target fail |
| v272d | v266c | boost-only learned head floor 1.00, ceiling 1.05 | 20.675602 / 0.720284 / 0.152049 | 19.843998 / 0.620177 / 0.179918 | +0.011944 / +0.000267 / +0.000417 | target fail |
| v272e | v270d | texture-lowrank boost-only learned head | 20.674818 / 0.720303 / 0.152075 | 19.844132 / 0.620207 / 0.179923 | +0.012078 / +0.000296 / +0.000412 | target fail |

## Interpretation

The implementation worked, and the learned head is not random: label/prediction correlations were about `0.32-0.34`, and all full v272 runs improved policy-val over v266c or v270d. However, the improvement did not transfer to target exact.

Important failure pattern:

- v272b/v272c shrink variants reduce target PSNR and LPIPS relative to v266c.
- v272d boost-only improves policy-val most strongly but still lowers target PSNR/SSIM/LPIPS frontier.
- v272e on the texture-lowrank bank is between v266c and v270d, but still not all-axis better than either reference.
- The texture-lowrank exact path is also substantially slower, so the cost is not justified by the current target gains.

Therefore v272 is a useful diagnostic and infrastructure upgrade, but not a quality breakthrough.

## Phase-J Gate

Phase-J flowers reference:

- PSNR `20.304358`
- SSIM `0.557770`
- LPIPS `0.329222`

All v272 target exact runs still fail the Phase-J PSNR gate by roughly `0.460` PSNR. No full9 run is allowed from this branch.

## Lesson

Policy-val supervised scalar confidence is not enough. It can overfit the policy-val split and still damage target exact, even when it only uses target-free features. The next representation attempt should not add another learned scalar head on the same carrier. It should change the residual carrier or supervision target more substantially:

- cross-UV coherent surface feature texture,
- view-dependent source-feature aggregation with no scalar-only bottleneck,
- patch/gradient residual objective with a certificate on target-visible support,
- or a target-free distribution-shift calibrator that can prove no-harm before exact target application.

Current status remains `NOT COMPLETE`.
