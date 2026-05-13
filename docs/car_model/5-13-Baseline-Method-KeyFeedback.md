# MeshSplatting Baseline Comparison and Method Status

Date: 2026-05-13

## Short Verdict

The current strongest endpoint should be described as a two-level system:

1. **Phase-J / MeshSplatOpt already beats the clean MeshSplatting baseline clearly on the selected full9 evidence set.**
   - Full9 strict RGB wins vs clean-best: `9 / 9`
   - Mean delta vs clean: `+1.331084 PSNR`, `+0.034702 SSIM`, `-0.063359 LPIPS`
   - Mean triangle reduction: `7.6479%`
   - Evidence: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`

2. **The newer Phase-S representation-level residual upgrade is now a real method change, but its extra gain over Phase-J is still too small.**
   - It has moved `bicycle` from repeated rejection to a passing train-val gate under the new render-calibrated subset path.
   - Current best hard-scene result: `bicycle top1 scale2`
     - train-val delta: `+0.000005722 PSNR`, `+0.000000238 SSIM`, `+0.000000238 LPIPS`
     - report-only test delta: `+0.000000000 PSNR`, `+0.000000000 SSIM`, `-0.000000030 LPIPS`
   - This is not yet a visually or statistically strong improvement. It is a stability breakthrough, not a paper-level margin.

Therefore: **relative to the most basic MeshSplatting baseline, the main Phase-J method is strong; relative to Phase-J, the latest Phase-S innovation is still early and low-amplitude.**

## Method in Plain Language

Basic MeshSplatting trains a scene representation and directly renders from it. It does not explicitly ask which triangles are unreliable, whether a local correction is safe, or whether a correction generalizes to held-out training views.

Our current method wraps MeshSplatting with a self-diagnosis and repair loop:

1. Train or reuse a MeshSplatting-style checkpoint.
2. Identify triangles whose rendered pixels repeatedly disagree with ground truth.
3. Estimate whether the residual is stable across multiple training views.
4. Apply only local, bounded corrections to reliable supports.
5. Re-render train-heldout views and accept the correction only if PSNR/SSIM/LPIPS do not regress.
6. Fall back to the previous Phase-J model when the correction fails.

The newest Phase-S change makes this more careful:

- It first writes a **candidate plan** of train-certified face-local residuals.
- It then materializes only a small subset of that plan.
- It tests different subset sizes and residual scales through the real render gate.
- This avoids the old failure where many locally plausible residual faces jointly caused LPIPS or balanced-score regression.

## Key Latest Findings

### Phase-J vs Clean MeshSplatting

Phase-J is currently the honest main endpoint for claims against the clean MeshSplatting baseline.

Example full9 rows:

| scene | clean-best | Phase-J | delta |
|---|---|---|---|
| bicycle | `23.302 / 0.660 / 0.332` | `24.022 / 0.702 / 0.266` | `+0.720 / +0.042 / -0.066` |
| flowers | `19.682 / 0.512 / 0.395` | `20.304 / 0.558 / 0.329` | `+0.622 / +0.046 / -0.065` |
| bonsai | `28.895 / 0.896 / 0.259` | `31.862 / 0.930 / 0.173` | `+2.967 / +0.034 / -0.087` |

This is the part of the project that currently supports a strong baseline-over-baseline story.

### Phase-S vs Phase-J

Phase-S is not yet a strong universal improvement over Phase-J.

Recent bicycle render-calibrated subset search:

| candidate | accepted | train-val balanced | report-only test balanced | reading |
|---|---:|---:|---:|---|
| top1 scale1 | yes | `+0.000005007` | `-0.000001788` | passes train, tiny negative test balanced |
| top1 scale2 | yes | `+0.000005722` | `+0.000000596` | current best safe point |
| top1 scale3 | yes | `+0.000010133` | `+0.000000000` | train stronger, test neutral |
| top1 scale4 | yes | `+0.000013828` | `-0.000005722` | train stronger but test negative |
| top2/top4/top7 | no | negative balanced | mixed | too many faces reintroduce LPIPS risk |

Evidence paths:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_bicycle/decisions/bicycle_decision.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_plan_20260513/bicycle/facelocal_sh3_candidate_plan.json`

### Cross-Scene Check

The same fixed `top1 scale2` protocol has now also completed on `flowers` and passed:

| scene | accepted | train-val balanced | report-only test balanced | faces / vertices added |
|---|---:|---:|---:|---:|
| bicycle | yes | `+0.000005722` | `+0.000000596` | `1 / 3` |
| flowers | yes | `+0.000010729` | `+0.000005007` | `1 / 3` |

Flowers evidence:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_flowers/decisions/flowers_decision.json`
- selected face: `1036117`
- materialization scale: `2.0`

Combined two-scene summary:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_combined/phasek_barycentric_gate_summary_collected.md`
- accepted scenes: `2 / 2`
- mean effective test delta: `+0.000002 PSNR`, `+0.000000 SSIM`, `-0.000000 LPIPS`

## Honest Weaknesses

- Phase-S gains are still near numerical noise, often `1e-6` to `1e-5`.
- Current visual improvement is unlikely to be obvious to a human viewer.
- The method can pass gate by being extremely conservative; that is reliable but not exciting enough.
- Adding more faces currently hurts balanced train-val because LPIPS is very sensitive.
- We still need a stronger carrier or better render-level objective to turn stable local residuals into visible gains.

## Next Required Work

1. Rerun the same fixed policy on more scenes without per-scene hand tuning.
2. If the multi-scene result remains tiny, upgrade the Phase-S carrier rather than continuing threshold games.
3. Add qualitative crops only after the quantitative margin is large enough to be visible.
4. Keep claims separated:
   - Phase-J: strong baseline-over-MeshSplatting claim.
   - Phase-S: real representation-level research direction, but not yet paper-final.

## Current Claim Boundary

Safe claim:

> MeshSplatOpt Phase-J substantially improves over the clean MeshSplatting baseline on the selected full9 evidence set, while preserving a compact representation. The newer Phase-S representation-level repair path introduces a stricter render-calibrated candidate-plan mechanism and has begun to stabilize hard scenes, but its current extra gains are not yet large enough for a final paper claim.

Unsafe claim:

> Phase-S fully solves representation-level repair or visually/quantitatively dominates Phase-J across scenes.
