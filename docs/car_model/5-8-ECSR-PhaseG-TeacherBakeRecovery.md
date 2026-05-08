# 5-8 ECSR Phase-G Teacher-Bake Recovery

## Goal

Phase-F alpha-grid ELA is the current strongest result, but it is still a render-time adapter. Phase-G tests whether that teacher can be baked back into a topology-frozen MeshSplatting checkpoint, so the final artifact is a normal representation-level model rather than a post-render correction pipeline.

This phase is therefore not another parameter sweep. The intended closed loop is:

1. Start from the fixed Phase-F compact checkpoint selected by policy validation.
2. Reuse the already selected Phase-F train-only ELA policy/alpha instead of reselecting per run.
3. Generate train-split ELA teacher renders.
4. Run topology-frozen recovery with teacher-render loss and parent-render rollback.
5. Render the final checkpoint on the same held-out split and compare against clean MeshSplatting, Phase-F source ELA, and Phase-F render-time alpha-grid ELA.
6. Audit that triangle/vertex counts remain unchanged during recovery.

## Implementation

New runner:

`scripts/car_model/ecsr_run_phaseg_teacher_bake_recovery.py`

Supporting ELA speed/fairness patch:

`scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`

The runner defaults to `--teacher_policy_source phasef_report`, which reads the fixed policy from the Phase-F ELA report:

`<phasef compact model>/test/ours_26000_phasef_alpha0875grid_ela/ela_report.json`

This avoids recomputing auto-policy selection during teacher generation and keeps the teacher fixed to the policy already used in the reported Phase-F result.

## Active Pilot Runs

Output roots:

- `outputs/carnet/meshsplatopt/ecsr_phase_g/teacher_bake_alpha0875_pilot_v1`
- `outputs/carnet/meshsplatopt/ecsr_phase_g/teacher_bake_alpha0875_pilot_strong_v1`

Pilot scenes:

- `bicycle`, default Phase-G recipe: `feature_lr=3e-5`, `teacher_lambda=0.05`, `teacher_dssim=0.10`, parent rollback `3.0`.
- `flowers`, stronger teacher recipe: `feature_lr=3e-5`, `teacher_lambda=0.08`, `teacher_dssim=0.12`, parent rollback `3.0`.

Both runs use W&B:

- Project: `mesh-splatting-ecsr`
- Groups:
  - `phase_g_teacher_bake_alpha0875_pilot_v1`
  - `phase_g_teacher_bake_alpha0875_pilot_strong_v1`

## Split Correction

The first pilot used the policy-validation file split for recovery and rendering. That is useful as a diagnostic split, but it is not the same as the Phase-F full9 / clean MeshSplatting reporting protocol. For example, bicycle has 34 policy-val test frames but 25 LLFF official test frames. Comparing those metrics directly is invalid.

The runner has therefore been corrected to default to official `llff` for both training and final evaluation:

- `--train_split_strategy llff`
- `--eval_split_strategy llff`

The `file` split remains available only for diagnostic policy-val checks. Paper-facing Phase-G results must use `llff` unless a table explicitly states otherwise.

New official pilot roots:

- `outputs/carnet/meshsplatopt/ecsr_phase_g/teacher_bake_alpha0875_official_pilot_v1`
- `outputs/carnet/meshsplatopt/ecsr_phase_g/teacher_bake_alpha0875_official_pilot_strong_v1`

## Acceptance Criteria

Minimum for continuing this branch:

- final checkpoint beats clean-best MeshSplatting on PSNR, SSIM, and LPIPS;
- topology remains unchanged from iteration 26000 to the final recovery iteration;
- no test-split ELA is used by the baked model;
- if the baked model cannot approach Phase-F render-time ELA, the limitation must be documented and the next method change should target representation capacity rather than more scalar tuning.

Stronger paper-facing target:

- baked checkpoint is competitive with or better than Phase-F render-time ELA;
- qualitative examples show visible improvement without post-render correction;
- the same fixed recovery recipe works across all selected scenes.

## Official Pilot Outcome

The official-split pilots were completed and rejected as a final direction.

| scene | recipe | final PSNR | final SSIM | final LPIPS | delta vs clean | delta vs Phase-F ELA | topology |
|---|---|---:|---:|---:|---:|---:|---|
| bicycle | default | 23.290770 | 0.659508 | 0.332462 | -0.010843 / -0.000359 / +0.000385 | -0.629343 / -0.036406 / +0.057079 | unchanged |
| flowers | strong teacher | 19.666090 | 0.511543 | 0.394951 | -0.016167 / -0.000279 / +0.000388 | -0.538599 / -0.038988 / +0.051967 | unchanged |

Interpretation:

- the topology-frozen checkpoint can absorb the teacher weakly, but not enough
  to beat clean MeshSplatting;
- it is far below the render-time Phase-F / Phase-H ELA teacher;
- increasing the teacher weight on `flowers` did not solve the issue;
- the limitation is representation capacity and locality, not merely training
  length.

Decision: do not continue Phase-G as the main line without a deeper
representation change. The accepted route after this result is the guarded
render-time portfolio in
[`5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md`](5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md),
while representation-level recovery remains a separate future research risk.
