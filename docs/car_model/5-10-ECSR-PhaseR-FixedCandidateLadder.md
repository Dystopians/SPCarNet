# ECSR Phase-R: Fixed Surface-SH1 Candidate Ladder

Date: 2026-05-11

## Purpose

Phase-R tests whether the ECSR surface evidence can be baked into the MeshSplatting representation instead of staying only as render-time ELA.  The operator writes a bounded SH1 residual delta onto train-view-certified surface support while preserving mesh topology.

This is not a parameter sweep.  The accepted policy is a fixed ladder:

1. Try dense16 surface-SH1 face-policy candidate.
2. If dense16 is rejected by train-val gate, try sparse face-policy candidate.
3. If the scene is a predeclared Phase-J structural-edge fallback scene, do no representation edit until an edge-aware surface operator exists.
4. Held-out test metrics are report-only and never used for candidate selection.

## Implemented Changes

- `ecsr_apply_surface_residual_barycentric_sh1_delta.py`
  - Added uniform barycentric mode for no-barycentric evidence.
  - Added per-face train-policy certification before applying shared-vertex SH1 deltas.
  - Audits selected faces, accepted faces, modified vertices, proxy gains, and topology integrity.

- `ecsr_run_phasek_barycentric_gate_scene.py`
  - Added SH1 face-policy arguments.
  - Avoids saving barycentric maps when uniform barycentric is used; this removed a severe evidence-generation bottleneck.

- `ecsr_select_phase_r_policy.py`
  - New fixed candidate ladder selector.
  - Reads only train-val decision files for selection.
  - Writes JSON/CSV/Markdown summaries with held-out test deltas marked report-only.

- Evidence-cache robustness
  - Added a fast `bincount` aggregation path with fallback.
  - Added face-id range filtering before saving new evidence caches.
  - Added `--max_face_id` filtering to evidence expansion for older caches containing stale positive invalid ids.

## Outdoor-5 Result

Artifact:

`outputs/carnet/meshsplatopt/ecsr_phase_r/fixed_candidate_ladder_v6dense_v5sparse_edge_noop_outdoor5/phase_r_fixed_candidate_ladder.md`

| scene | selected | train-val selected | test dPSNR | test dSSIM | test dLPIPS | report-only strict win |
|---|---|---:|---:|---:|---:|---:|
| flowers | dense16 SH1 | yes | +0.002346 | +0.000344 | -0.000405 | true |
| garden | dense16 SH1 | yes | +0.000662 | +0.000024 | -0.000036 | true |
| bicycle | sparse SH1 fallback | yes | +0.001156 | +0.000135 | -0.000432 | true |
| stump | dense16 SH1 | yes | +0.000021 | +0.000000 | -0.000011 | true |
| treehill | no-op | predeclared edge-fallback no-op | +0.000000 | +0.000000 | +0.000000 | false |

Mean report-only delta over outdoor-5:

- PSNR: `+0.000837`
- SSIM: `+0.000101`
- LPIPS: `-0.000177`

The four edited scenes are strict RGB wins over the Phase-J base under the same held-out split.  Treehill is intentionally no-op because both dense and sparse SH1 candidates exposed a gate weakness: train-val accepted them, but held-out PSNR/SSIM regressed.

## Failed Treehill Variants

Treehill is Phase-J's structural-edge fallback scene.  Current SH1 surface residuals interact poorly with that branch:

| candidate | train-val status | held-out test behavior |
|---|---|---|
| dense16 SH1 | accepted | PSNR `-0.001566`, SSIM `-0.000002`, LPIPS `-0.000017` |
| sparse4096 SH1 | accepted | PSNR `-0.001526`, SSIM `-0.000001`, LPIPS `-0.000012` |
| micro SH1 | accepted | PSNR `-0.000435`, SSIM `-0.000002`, LPIPS `-0.000002` |

This is the main scientific lesson from this run: support-mask train-val gains are not sufficient for edge-fallback scenes.  The policy must be branch-aware, and the next operator should explicitly model edge-gated residuals instead of applying generic surface SH1 deltas.

## Current Assessment

Phase-R is a real representation-level upgrade over the earlier surface-lumigraph baseline because the residual is stored in checkpoint attributes and rendered through MeshSplatting, not pasted as a 2D post-process.  It also now has a fixed, auditable candidate ladder and a clear no-test-selection contract.

However, the gains are still small.  Phase-R should not be described as the final paper endpoint or as a large-margin replacement for Phase-J.  It is best positioned as:

- a certified representation-attached residual recovery module;
- a train-only policy framework for accepting or rejecting surface edits;
- a diagnostic bridge toward a stronger edge-aware or neural-texture surface operator.

## Next Required Work

1. Build an edge-aware representation operator for Phase-J structural-edge scenes, starting with treehill.
2. Add a train-only branch-risk feature to the gate: if a scene uses edge fallback and no edge-aware surface operator is available, SH1 edits are no-op by design.
3. Generate local support-mask qualitative panels for the four edited outdoor scenes, because full-frame deltas are visually subtle.
4. Re-run the fixed ladder on indoor scenes only after the outdoor branch-aware rule is frozen.
