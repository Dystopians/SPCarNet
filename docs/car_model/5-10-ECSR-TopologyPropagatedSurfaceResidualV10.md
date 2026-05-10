# ECSR Topology-Propagated Surface Residual V10

Date: 2026-05-10

This pass extends the V8 surface residual lumigraph into a topology-aware,
train-gated representation path. It is not promoted as the final paper method:
it fixes several real interface and policy weaknesses, but the held-out gains
remain tiny and not yet competitive with the strongest render-time Phase-J
result.

## What Changed

V9/V10 keeps the residual attached to persistent rendered `face_id`, but no
longer requires the target pixel to hit the exact same face that carried train
residual evidence.

The new path is:

1. Fit train residual observations on high-confidence source faces.
2. Load compact checkpoint topology from `_triangle_indices`.
3. Build a topology alias from residual source faces to actually visible
   neighboring target faces.
4. Calibrate a global alpha on train-policy views only.
5. Fit a per-face train gate.
6. Intersect primary and consensus face gates before writing held-out renders.

V10 adds a guard-aware policy correction: calibration now chooses the best row
among candidates that satisfy the metric guards, instead of choosing the raw
highest score and rejecting the whole method if that one row violates SSIM. It
also supports provisional face-gate recalibration: when the raw signal has
positive PSNR but fails a structure guard, a provisional train-only face gate is
fit first, and calibration is rerun on the gated signal.

## Engineering Fixes

Several bottlenecks were fixed because they made full-scene validation
impractical:

- `.npz` evidence loading now extracts cached `.npy` members and reads them via
  `open_memmap`; direct `np.load(.npy)` was unexpectedly slow in this
  environment.
- Candidate `face_id` membership now filters to checkpoint-valid ids and uses
  bounded table lookup where possible.
- Topology propagation is restricted to train/test actually visible faces,
  avoiding full scans over 11M+ checkpoint triangles.
- `compute_surface_signal` groups active pixels by face id and caches per-source
  residuals per target camera.
- `fit_face_gate` now groups active pixels by face id instead of scanning the
  whole image once per candidate face.
- LPIPS model construction is cached, although V10 policy runs used PSNR/SSIM
  calibration for speed; held-out evaluation still reports LPIPS independently.

## Held-Out Diagnostics

All numbers below are independent `evaluate_render_split_metrics.py` outputs on
held-out test renders, compared to `ours_26000_phasef_extra_compact_base`.

| scene | method | accepted faces | coverage | dPSNR | dSSIM | dLPIPS | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| garden | V9 toporing1 gate128 tol | 26 | 0.058% | +0.000744 | +0.000001 | -0.000006 | positive but tiny |
| garden | V9 toporing1 gate32 tol | 449 | 0.188% | +0.001562 | +0.000001 | -0.000014 | best V9 garden diagnostic |
| flowers | V10 guard-aware gate32 | 320 | 0.190% | +0.000769 | -0.000010 | -0.000010 | mixed; not promoted |

Raw metric references:

- `outputs/carnet/meshsplatopt/ecsr_phase_l/v9_toporing1_garden_gate_sweep_eval.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_l/v10_guardaware_flowers_eval.json`

W&B runs:

- garden gate128 tol: `hkfdffs0`
- garden gate32 tol: `q58r746t`
- flowers guard-aware gate32: `xbcvfuzt`

## Interpretation

The important result is not the absolute metric gain. The important result is
the diagnosis:

- Topology propagation can expand the candidate support by about an order of
  magnitude.
- Train-only face gating is necessary; ungated propagation improves PSNR but
  can hurt SSIM.
- Guard-aware candidate selection is required; the previous calibrator could
  reject a valid lower-alpha candidate because a higher-scoring alpha violated
  SSIM.
- Even after these fixes, final held-out coverage remains below 0.2% on the
  tested outdoor scenes, so the effect is visually and numerically marginal.

The current bottleneck is therefore residual expressivity and support, not just
policy plumbing. A per-face constant residual is too weak. The next credible
research step should replace the constant residual with a compact local
appearance model, for example a vertex/face low-rank residual field, a
view-conditioned surface basis, or a small train-only residual MLP over
surface-id, barycentric coordinate, normal, and view direction. The V10 policy
and gate stack can then serve as the safety certificate for that higher-capacity
field.

## Status

Keep V10 as an implementation milestone and diagnostic baseline. Do not claim
that it solves the paper objective. It provides a cleaner representation-level
path than V8, but it still does not deliver a large, visually obvious,
cross-scene gain over MeshSplatting or Phase-J.
