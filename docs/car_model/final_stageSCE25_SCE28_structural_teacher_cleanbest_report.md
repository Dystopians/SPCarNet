# Final Stage SCE25-SCE28 Structural / Teacher / Clean-Best Report

Date: 2026-05-06

Decision: `SCE27_BONSAI_STILL_MIXED_SCE28_CLEANBEST_RESET_BLOCKED_BY_LARGE_CHECKPOINT_IO`

## Goal

The previous SCE23/SCE24 line made the policy safer but did not improve bonsai over the F82 parent on all metrics. This stage tested three non-per-scene method upgrades:

1. structural one-sided parent render rollback with local DSSIM and Sobel edge evidence;
2. certified teacher-student recovery using a high-fidelity clean teacher on train views only;
3. a clean-best reset that starts compaction from the much stronger clean 9000 checkpoint rather than the degraded clean 22000/F82 lineage.

The design is motivated by evidence-constrained reconstruction rather than parameter search: sparse SfM evidence is treated as a geometry certificate, parent render evidence becomes an appearance non-regression certificate, and teacher render supervision is only active where the teacher is closer to train GT.

Related foundations:

- 3D Gaussian Splatting: https://arxiv.org/abs/2308.04079
- DS-NeRF sparse depth evidence: https://arxiv.org/abs/2107.02791
- Conformal risk control / certificate framing: https://arxiv.org/abs/2208.02814
- CVaR tail-risk objective: https://doi.org/10.21314/JOR.2000.038

## Implementation

Code changes:

- `train.py`: added differentiable local DSSIM and Sobel edge maps to parent render rollback.
- `arguments/__init__.py`: added opt-in structural rollback knobs.
- `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py`: exposed `l1_dssim`, `l1_edge`, and `l1_dssim_edge` parent rollback modes.
- `scripts/car_model/meshsplatopt_run_sce_policy_recovery.py` and `utils/sce_recovery_policy.py`: propagated the same policy controls.
- `ss3dm_prior/meshsplatopt/checkpoint_compaction.py`: added opt-in `keep_unused_vertices` face-only compaction for very large checkpoints.
- `ss3dm_prior/meshsplatopt/compact_selector.py`: reduced large-area memory pressure and switched large top-k selection to `argpartition`.
- `scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py`: exposed `--keep_unused_vertices`.

All new training behavior is opt-in.

## Experiments

Parent for SCE25-SCE27:

`outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/bonsai/adaptive_global_policy_v5_seed0/recovery_model`, iteration 26000.

| run | W&B | mechanism | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SCE25 | `gr8jx8ud` | structural `l1_dssim_edge` ATR + CTR-SCE | +0.000994 | -0.000914 | +0.000461 | -0.000197 | -0.002802 | +0.002458 | rejected |
| SCE26 | `lqh3v4m7` | clean9000 train teacher + structural ATR + CTR-SCE + LPIPS | +0.000938 | -0.000915 | +0.000094 | -0.000135 | -0.001980 | +0.001421 | rejected |
| SCE27 | `qyr6n1gs` | appearance-only clean9000 train teacher with geometry LR zero | +0.001031 | -0.000828 | +0.000086 | -0.000008 | -0.000025 | +0.020408 | rejected |

All three runs keep triangle topology unchanged, use W&B online logging, and are evaluated by independent render metrics plus `evaluate_geometry_colmap.py`.

## Interpretation

Structural ATR helped depth but did not fix the appearance bottleneck: SSIM and LPIPS still regress on bonsai. Adding a clean 9000 train teacher reduced the LPIPS regression from `+0.000461` to `+0.000094`, but SSIM remained negative. The appearance-only attempt confirms that simply freezing geometry is not sufficient; render weights can still affect the COLMAP surface proxy and normal metric.

The important lesson is that bonsai is not just a rollback-strength problem. The parent lies on a very narrow tradeoff: tiny updates can improve PSNR/depth while hurting structural/perceptual metrics. A reliable method must either no-op to the parent or start from a better clean checkpoint.

## Clean-Best Reset Finding

The existing clean bonsai checkpoint at iteration 9000 has much stronger render metrics than the clean 22000/F82 lineage:

- clean 9000: PSNR `18.541124`, SSIM `0.463496`, LPIPS `0.483265`
- F82 parent 26000: PSNR `11.069180`, SSIM `0.241154`, LPIPS `0.572932`

This means future paper-facing claims must not compare only to clean 22000 if clean 9000 is the best clean baseline. SCE28 began implementing a clean-best reset, but the 9000 checkpoint is very large: `2,487,474` triangles and `2,478,890` vertices. Generic CSEF compaction and direct low-importance candidate extraction both exposed a large CPU/checkpoint bottleneck before a full run could complete in this stage.

The code now has the required fast face-only compaction hook, but SCE28 still needs a cached low-evidence selector artifact or a streaming checkpoint reader before it can be run end-to-end.

## Verification

- `python -m py_compile` passed for modified training, wrapper, policy, compaction, and selector files.
- `scripts/car_model/smoke_test_stageSCE23_parent_render_tail_rollback.py` passed, including the new structural rollback branch.
- `scripts/car_model/smoke_test_stageSCE12_evidence_conflict_graph.py` passed.
- `scripts/car_model/smoke_test_stageSCE13_certificate_edit_planner.py` passed.
- `scripts/car_model/smoke_test_stageSCE14_stress_test_defects.py` passed.
- A synthetic `keep_unused_vertices` compaction smoke test passed.

The legacy full F5 compaction smoke was intentionally stopped because it runs a large parking compaction and is not a quick smoke test.

## Next Decision

Do not claim bonsai superiority over F82 from SCE25-SCE27. The correct next step is:

1. build a cached/streaming low-evidence selector for clean-best 9000 checkpoints;
2. run clean-best 9000 -> compact -> short recovery with the same independent metrics;
3. compare against the best clean baseline, not only clean 22000;
4. if clean-best compaction preserves most of the 9000 quality, promote SCE28 as the new fair baseline-reset route.

