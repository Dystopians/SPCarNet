# ECSR Surface FaceAlpha V11-V13

Date: 2026-05-10

This note records the follow-up after topology-propagated surface residual V10.
The goal was to make the representation-attached residual stronger without
falling back to scene-specific parameter search. The result is a useful method
upgrade and a sharper bottleneck diagnosis, but it is still not the final paper
closed loop.

## V11: Per-Face Alpha Surface Residual

V11 replaces the previous global-alpha surface residual with a train-fitted
per-face alpha. Each candidate surface face receives its own ridge-regularized
residual strength from train-policy views, with shrinkage for low-support faces.
The final held-out field is the intersection of the primary policy split and a
consensus split; target views use only rendered surface maps and no held-out GT.

Implementation:

- `scripts/car_model/ecsr_apply_surface_residual_facealpha_adapter.py`

Policy settings used for the first validation:

- `policy_val_stride=4`, consensus stride `2`;
- topology ring `1`, `neighbor_max_targets_per_source=64`;
- `face_alpha_max=0.25`, `face_alpha_min_pixels=32`;
- W&B group: `phase_p_surface_facealpha_v11`.

Independent held-out results against `ours_26000_phasef_extra_compact_base`:

| scene | method | accepted faces | target coverage | dPSNR | dSSIM | dLPIPS | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| garden | V11 face-alpha | 493 | 0.2020% | +0.001804 | +0.000006 | -0.000017 | strict positive, best surface-only garden row |
| flowers | V11 face-alpha | 350 | 0.2059% | +0.000898 | -0.000006 | -0.000021 | mixed; improves V10 but still loses SSIM |

Raw metrics:

- `outputs/carnet/meshsplatopt/ecsr_phase_l/v11_facealpha_garden_eval.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_l/v11_facealpha_flowers_eval.json`

W&B runs:

- garden: `ceule6xm`
- flowers: `5dqh1jrf`

## V12: Edge-Aware Alpha Fitting

V12 adds a train-only local-gradient objective to the face-alpha fit. The intent
is to move structure risk into the per-face decision instead of only checking a
scene-level SSIM guard after fitting.

Two engineering changes were required:

- face aggregation was changed from repeated full-image sorts to dense
  `bincount` accumulation;
- the edge objective now uses deterministic pixel stride sampling
  (`face_alpha_edge_stride=16` in the validated run), which keeps the method
  reproducible and makes multi-scene runs practical.

After the flowers/garden runs, treehill was used as a large-scene stress test.
The checkpoint has `8,402,362` faces and the target surface-map directory has
18 views. A first dict-based V11 run was terminated after about 20 minutes in
CPU fitting. A second dense-accumulation run passed primary and consensus
surface-signal preparation but was still CPU-bound after about 49 minutes before
target output. Therefore treehill is not counted as a completed metric row. The
lesson is that dense accumulation fixes small/medium scenes but the current
face-alpha fitter still needs candidate-level sparsification or GPU/batched
aggregation before it can be treated as a full9 routine.

Flowers held-out result:

| scene | method | edge weight | edge stride | dPSNR | dSSIM | dLPIPS | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| flowers | V12 edge-aware face-alpha | 0.5 | 16 | +0.000904 | -0.000006 | -0.000021 | tiny PSNR/LPIPS improvement; SSIM still mixed |

Raw metrics:

- `outputs/carnet/meshsplatopt/ecsr_phase_l/v12_facealpha_edge_flowers_eval.json`

W&B run:

- flowers: `tw3m5sw0`

## V13 Diagnostics: Why Surface-Only Is Not Enough

The remaining flowers SSIM loss is extremely small, but it is consistent:
surface-only residual improves PSNR/LPIPS while introducing a `1e-6`-scale
structure penalty. I tested two fixed, non-GT inference guards on the existing
V11 residual:

1. luminance/chroma decomposition;
2. low-gradient application masks from the base render.

The best diagnostic rows almost eliminate the SSIM loss but also reduce the
useful residual:

- `luma0.2`: PSNR and LPIPS stay positive, SSIM is only about `1.2e-7` below
  base;
- low-gradient masks keep LPIPS positive, but do not make SSIM strictly
  positive.

This says the current surface residual is already near the numerical boundary
on flowers; the real weakness is coverage and representational capacity, not a
missing scalar guard.

## ELA Teacher Comparison

Existing guarded ELA remains much stronger on the same flowers compact base:

| method | PSNR | SSIM | LPIPS | dPSNR vs base | dSSIM vs base | dLPIPS vs base |
|---|---:|---:|---:|---:|---:|---:|
| base | 19.668695 | 0.511678 | 0.394788 | 0.000000 | 0.000000 | 0.000000 |
| V11 surface face-alpha | 19.669594 | 0.511672 | 0.394767 | +0.000898 | -0.000006 | -0.000021 |
| Phase-J guarded ELA | 20.304358 | 0.557770 | 0.329222 | +0.635662 | +0.046092 | -0.065565 |
| Phase-J ELA + V11 surface | 20.305147 | 0.557764 | 0.329209 | +0.636452 | +0.046086 | -0.065578 |
| Phase-J ELA + luma0.2 surface | 20.304554 | 0.557770 | 0.329222 | +0.635859 | +0.046092 | -0.065566 |

Raw metrics:

- `outputs/carnet/meshsplatopt/ecsr_phase_l/flowers_existing_ela_vs_surface_eval.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_l/flowers_ela_surface_combo_eval.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_l/flowers_ela_surface_luma_combo_eval.json`

The important conclusion is that surface-attached residuals can add tiny
PSNR/LPIPS improvements on top of ELA, but they currently do not provide the
large visual jump. Phase-J remains the reliable result to report for RGB
quality; V11/V12 is a representation-level research component that still needs
larger support and stronger local appearance modeling.

## Current Decision

Do not promote V11/V12 as the final method. Keep it as the safest
representation-attached residual baseline so far:

- V11 is strictly better than V9/V10 on garden.
- V11/V12 improve flowers PSNR/LPIPS but still miss strict SSIM.
- The accepted target coverage remains about `0.2%`, too small for visible
  qualitative improvement.
- Large-scene fitting is not yet scalable enough for routine full9 validation;
  treehill exposed this despite dense accumulation.
- ELA gives the actual large RGB improvement and should remain the teacher or
  fallback policy until its gains are distilled into a persistent surface code.

The next credible upgrade should not be another scalar alpha search. It should
increase surface support and expressivity, for example:

- a view-conditioned local surface residual basis;
- barycentric or vertex-attached residual features instead of per-face constants;
- an ELA-to-surface distillation stage with train-policy Pareto guards;
- a portfolio policy where ELA is the teacher/fallback and surface code is
  accepted only when it gives strict train-policy gains over the teacher.

## V14: Policy-Scoped Alias And Compact Alpha Fit

V14 fixes the large-scene scalability problem exposed by treehill without
changing the residual semantics.

Two changes were made:

1. topology alias candidates default to train-policy visible faces
   (`--alias_candidate_scope policy`) instead of train+held-out target visible
   faces. This is stricter and faster: a face can receive an alpha only if it is
   certified on train-policy views, so target-only visibility should not enlarge
   the policy alias set;
2. `fit_face_alphas` now remaps active policy faces to a compact index space
   before accumulation. On treehill, primary alpha fitting dropped from about
   `285s` to `0.06s`.

The policy-eval path was also changed to dense alpha lookup rather than
per-face Python assignment. This made primary evaluation on treehill about
`1.5s`; consensus evaluation still has expensive large-view surface-signal
costs, but the run is now complete instead of stalled.

Held-out V14 results:

| scene | V14 behavior | PSNR | SSIM | LPIPS | verdict |
|---|---|---:|---:|---:|---|
| garden | accepted, same output as V11 | 25.029341 | 0.780037 | 0.201304 | strict positive vs base |
| flowers | accepted, same output as V11 | 19.669594 | 0.511672 | 0.394767 | PSNR/LPIPS positive, SSIM still slightly negative |
| treehill | rejected by consensus, no-op output | 20.923227 | 0.564224 | 0.406108 | no negative transfer |

Raw metrics:

- `outputs/carnet/meshsplatopt/ecsr_phase_l/v14_policyalias_garden_eval.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_l/v14_policyalias_flowers_eval.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_l/v14_policyalias_treehill_eval.json`

W&B runs:

- treehill: `2aiaift2`
- garden: `1gy499q8`
- flowers: `tanumx5u`

V14 changes the status from "not scalable enough for treehill" to "scalable
enough to complete and reject risky scenes." It still does not solve the core
paper weakness: coverage remains around `0.2%` on accepted outdoor scenes, and
flowers still lacks strict SSIM improvement. The next method step should be
capacity/coverage, not more alpha plumbing.
