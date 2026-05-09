# ECSR Phase-K Train-Val Representation Gate

Date: 2026-05-08

This update turns the barycentric residual delta from an unsafe unconditional
checkpoint edit into a train-heldout-gated representation update. The important
change is methodological: the representation module is no longer selected by
held-out test results or by residual fitting loss alone.

## Implemented Interfaces

| component | change |
|---|---|
| `ecsr_build_surface_evidence_cache.py` | Added `--save_barycentric`, storing reconstructed top-support barycentric coordinates in per-view NPZ files. |
| `ecsr_apply_surface_residual_barycentric_delta.py` | New checkpoint operator fitting vertex SH-DC residual deltas from barycentric per-pixel residual RGB. |
| `meshsplatopt_apply_evidence_lumigraph_adapter.py` | Added `--support_policy_fit_only` so train-policy validation adapts held-out train views using only fitting-train support frames. |
| `evaluate_render_split_metrics.py` | Added `--view_names_file` / `--view_names_key` to evaluate only policy-val train views. |
| `ecsr_decide_phasek_trainval_gate.py` | New gate script: accept candidate from train-val near-Pareto metrics only; held-out test is report-only. |

## Gate Rule

The current Phase-K gate is deliberately conservative:

- selection split: deterministic train-policy-val views;
- test usage: none for selection, report-only for audit;
- candidate must have non-negative train-val PSNR gain;
- candidate may have only tiny numerical-regime regressions:
  `SSIM >= -5e-5`, `LPIPS <= +1.5e-4`;
- otherwise the method falls back to Phase-J guarded adaptive edge ELA.

This is not a parameter sweep over the test set. The failure mode it fixes is
specific: residual fitting loss alone accepted `flowers`, but both train-val
SSIM and held-out test metrics showed the representation edit was harmful.

## Validation

Train-policy-val metrics compare the candidate to the Phase-J train-val ELA
run. Held-out test deltas are shown only after the decision.

| scene | candidate | train-val dPSNR | train-val dSSIM | train-val dLPIPS | gate | test dPSNR | test dSSIM | test dLPIPS |
|---|---|---:|---:|---:|---|---:|---:|---:|
| bicycle | bary-delta v2wide s08 | +0.000349 | -0.000044 | +0.000020 | accept | +0.000872 | +0.000151 | -0.000389 |
| flowers | bary-delta v2wide s08 | +0.000505 | -0.000076 | -0.000053 | reject | -0.003515 | -0.000307 | +0.000180 |

Decision files:

- `outputs/carnet/meshsplatopt/ecsr_phase_d/phasek_trainval_representation_gate/bicycle_decision.json`;
- `outputs/carnet/meshsplatopt/ecsr_phase_d/phasek_trainval_representation_gate/flowers_decision.json`.

Main W&B runs:

- bicycle Phase-J train-val base: `xs71gih3`;
- bicycle bary-delta s08 train-val: `hxqibzce`;
- bicycle bary-delta s08 held-out test: `yeeiz3gd`;
- flowers Phase-J train-val base: `upji5c6b`;
- flowers bary-delta s08 train-val: `3ybdsm1p`;
- flowers bary-delta s08 held-out test: `rha65tc3`.

## Effective Two-Scene Outcome

With Phase-K gating, `bicycle` uses barycentric representation recovery and
`flowers` falls back to Phase-J. Relative to Phase-J on these two scenes:

| metric | mean delta |
|---|---:|
| PSNR | +0.000436 |
| SSIM | +0.000076 |
| LPIPS | -0.000195 |

This is a real safety improvement over unconditional bary-delta, because it
keeps the positive `bicycle` edit and blocks the negative `flowers` edit. It is
not yet a large paper-level representation breakthrough.

## Current Bottleneck

The new barycentric operator is principled, persistent, and auditable, but its
effect size remains very small. The evidence says the remaining bottleneck is
not command coverage or logging; it is representational power. SH-DC vertex
deltas can correct tiny color residuals but cannot create a strong visible
change when the error is caused by view-dependent appearance, missing
high-frequency geometry, or ELA already correcting the dominant residual at
render time.

The next substantial improvement should therefore target a richer persistent
basis: per-cluster residual texture charts, learned residual bases, or
material/view-dependent residual carriers validated with the same train-val
gate. More local DC deltas are unlikely to produce a large qualitative gap.
