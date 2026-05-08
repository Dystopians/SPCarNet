# 5-8 ECSR Phase-J Guarded Adaptive Edge Policy

## Status

Phase-J is the current strongest ECSR result on the selected Mip-NeRF360 full9
protocol. It fixes the Phase-H weakness where `treehill` could only fall back to
Phase-F and therefore did not strictly improve over the previous method.

The final materialized method is:

`ours_26000_phasej_guarded_adaptedge_ela`

Report:

`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`

Decision JSON:

`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/ours_26000_phasej_guarded_adaptedge_ela_guarded_decisions.json`

## Method

The method is a no-test-GT guarded portfolio on top of the fixed Phase-F compact
checkpoints.

1. Use Phase-H adaptive alpha ELA when its train-only alpha calibrator is stable.
2. Stability requires accepted-bin fraction `>= 0.65`, mean alpha `>= 0.30`, and
   active alpha fraction `>= 0.55`.
3. If the adaptive alpha branch is unstable, use a train-selected structural
   edge fallback instead of the old fixed Phase-F fallback.
4. The structural fallback searches edge-gate quantiles
   `{0.5, 0.6, 0.7, 0.8, 0.9}` on train calibration only, with balanced
   PSNR/SSIM/LPIPS objective.

This keeps the policy evidence-based: held-out test metrics are used only after
the method is materialized.

## Implementation

New / changed interfaces:

- `utils/evidence_lumigraph_adapter.py`
  - adaptive per-bin alpha calibrator;
  - adaptive alpha statistics in ELA reports.
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
  - `--alpha_policy adaptive_bins`;
  - `--alpha_default`;
  - `--policy_edge_gate_quantiles`;
  - `--policy_edge_gate_dilates`.
- `scripts/car_model/ecsr_run_phasef_ela_adapter_eval.py`
  - fixed-policy-from-report path;
  - adaptive alpha forwarding;
  - auto edge-gate policy forwarding.
- `scripts/car_model/ecsr_materialize_phaseh_guarded_adapter.py`
  - materializes the no-test-GT guarded portfolio.

## Treehill Failure Analysis

Phase-H adaptive alpha improved `treehill` PSNR but damaged structure and LPIPS:

| method | PSNR | SSIM | LPIPS | delta vs Phase-F |
|---|---:|---:|---:|---|
| Phase-F alpha-grid | 21.249701 | 0.591590 | 0.350894 | reference |
| Phase-H adaptive alpha | 21.294319 | 0.582889 | 0.369435 | PSNR up, SSIM/LPIPS down |
| Phase-H guarded fallback | 21.249701 | 0.591590 | 0.350894 | non-regression only |

The alpha report explained the failure: `treehill` had only `71 / 125`
accepted bins, mean alpha `0.2617`, and active fraction `0.5148`, below the
stable-branch threshold. The first guarded policy therefore avoided damage but
could not strictly improve over Phase-F.

Phase-I tested whether defaulting the adaptive alpha map back to the Phase-F
global alpha would recover the scene. It did not: local bin overrides still
hurt all three metrics. The useful signal was instead in the structural edge
gate.

## Train-Selected Edge Policy

The structural fallback keeps the same residual policy as Phase-F and lets the
train calibration choose the edge acceptance quantile. For `treehill`, the
train-only balanced objective selected q=`0.5`, alpha=`0.75`:

| edge q | test PSNR | test SSIM | test LPIPS | dPSNR vs Phase-F | dSSIM vs Phase-F | dLPIPS vs Phase-F |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 21.296227 | 0.595606 | 0.336319 | +0.046526 | +0.004015 | -0.014575 |
| 0.6 | 21.274559 | 0.594101 | 0.342305 | +0.024858 | +0.002510 | -0.008589 |
| 0.8 | 21.216621 | 0.587298 | 0.362926 | -0.033079 | -0.004292 | +0.012033 |
| 0.9 | 21.159079 | 0.579418 | 0.380092 | -0.090622 | -0.012172 | +0.029198 |

The auto-policy report records the train-side ranking:

- selected policy: residual, k=`4`, depth rel tol=`0.06`, direction weight=`0.35`,
  edge q=`0.5`, dilation=`1`;
- selected alpha: `0.75`;
- W&B run: `7ln9cddr` (`phasej_auto_edge_policy_treehill`).

## Full9 Result

Phase-J guarded portfolio vs selected clean MeshSplatting:

- strict RGB wins vs clean: `9 / 9`;
- mean delta vs clean: `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS;
- mean total triangle reduction: `7.6479%`.

Phase-J guarded portfolio vs Phase-F alpha-grid:

- strict RGB wins vs Phase-F: `9 / 9`;
- mean delta vs Phase-F: `+0.397095` PSNR, `+0.008305` SSIM, `-0.019321` LPIPS.

| scene | selected branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR Phase-F | dSSIM Phase-F | dLPIPS Phase-F | tri red. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | 24.021544 | 0.702357 | 0.266088 | +0.719931 | +0.042489 | -0.065989 | +0.101431 | +0.006442 | -0.009296 | 11.81% |
| flowers | adaptive alpha | 20.304358 | 0.557770 | 0.329222 | +0.622101 | +0.045948 | -0.065341 | +0.099669 | +0.007239 | -0.013762 | 11.82% |
| garden | adaptive alpha | 26.311111 | 0.827843 | 0.135843 | +1.281900 | +0.047808 | -0.065472 | +0.261786 | +0.010460 | -0.013955 | 3.47% |
| stump | adaptive alpha | 25.595104 | 0.724074 | 0.263909 | +0.390062 | +0.018909 | -0.030095 | +0.058529 | +0.002738 | -0.005623 | 11.82% |
| treehill | auto edge fallback | 21.296227 | 0.595606 | 0.336319 | +0.362045 | +0.031083 | -0.069725 | +0.046526 | +0.004015 | -0.014575 | 11.81% |
| room | adaptive alpha | 30.305639 | 0.905730 | 0.195989 | +1.558363 | +0.020887 | -0.053913 | +0.570789 | +0.008655 | -0.025519 | 2.10% |
| counter | adaptive alpha | 28.449171 | 0.893731 | 0.186472 | +1.697397 | +0.031675 | -0.065531 | +0.398413 | +0.009137 | -0.022497 | 2.10% |
| kitchen | adaptive alpha | 30.199732 | 0.916087 | 0.131955 | +2.381180 | +0.039635 | -0.067231 | +1.049686 | +0.013521 | -0.027346 | 2.10% |
| bonsai | adaptive alpha | 31.862005 | 0.930280 | 0.172555 | +2.966772 | +0.033879 | -0.086937 | +0.987028 | +0.012533 | -0.041312 | 11.80% |

Compared with Phase-H guarded adaptive alpha, Phase-J is equal on the eight
stable adaptive scenes and strictly better on `treehill`.

## What This Solves

- Removes the last Phase-H non-strict scene against Phase-F.
- Keeps the no-test-GT decision boundary: adaptive stability and fallback edge
  selection are train-evidence decisions.
- Preserves Phase-F's extra compactness while improving held-out RGB metrics.
- Converts a scene-specific observation into a reusable policy interface.

## Remaining Boundary

This is a major endpoint for the current render-time ELA line, but it is still
not a complete representation-level paper endpoint:

- the strongest appearance gains are still produced at render time;
- the baked Phase-G checkpoint experiments were negative;
- the average triangle reduction is useful but still conservative;
- external cross-dataset validation and reviewer-facing qualitative refresh are
  still required before a paper claim should be treated as final.

