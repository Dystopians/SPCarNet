# MeshPrior Stage 17 Real Variant Design

Date: 2026-05-01

## Goal

Build the first real `parking_phone_tiny` MeshPrior scene-optimization variant at the same 2000-iteration budget used by the medium baselines.

## Chosen Minimal Variant

Stage 17 uses the already gated MeshPrior checkpoint-copy cleanup as a conservative intervention at iteration `200`, then resumes normal scene optimization to iteration `2000`.

The pipeline is:

1. Start from the current-branch `baseline_200iter` checkpoint.
2. Use accepted copied-patch cleanup proposals that passed the no-op/floater/protect gate.
3. Apply those accepted cleanup edits only to a copied checkpoint.
4. Prepare a normal model directory containing that copied checkpoint at `point_cloud/iteration_200/`.
5. Resume current-branch training with `train.py --load_iteration 200 --iterations 2000`.
6. Evaluate with the same `render.py + metrics.py` and `evaluate_geometry_colmap.py` scripts used by the baselines.

This is a real training variant because the MeshPrior-edited checkpoint is not only evaluated offline; it becomes the initialization for continued optimization.

## Safety Rules

- The source baseline checkpoint is not overwritten.
- Proposal selection uses only inference-time scene evidence and the existing gate reports.
- No oracle ground truth is used to choose proposals.
- Rollback snapshots from the copied-patch proposal tests remain part of the audit trail.
- Final cleanup is disabled by default unless explicitly enabled and separately reported.
- The final report must include topology counts, final-cleanup state, W&B URL, render metrics, geometry proxy metrics, and proposal accept/reject counts.

## Outputs

- initialization/model root: `outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model`
- smoke root: `outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_smoke/model`
- W&B project: `spcarnet_meshprior`
- W&B group: `parking_stage17_real_variant`

## Gate

Stage gate is `PASS` only if the 2000-iteration run completes with online training-time W&B, checkpoint, render metrics, COLMAP geometry proxy, topology counts, and no unguarded geometry edit.

Stage gate is `SOFT PASS` if the variant is stable but does not improve over baselines.

Stage gate is `FAIL` if W&B/logs/checkpoints are missing, training crashes, or final cleanup unexpectedly edits topology.
