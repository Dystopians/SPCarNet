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

## Outdoor-5 Extension

On 2026-05-09, the same fixed Phase-K policy was run on the remaining outdoor
Mip-NeRF 360 scenes already present in the selected validation set:
`garden`, `stump`, and `treehill`. The runner is
`scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`; it performs the
full chain for each scene:

1. build barycentric rich surface evidence on the Phase-J selected compact
   checkpoint;
2. fit the fixed `bary_delta_v2wide_s08` checkpoint delta;
3. render candidate train/test evidence maps;
4. run W&B-logged ELA for train-policy-val and held-out test;
5. decide with `ecsr_decide_phasek_trainval_gate.py`.

The aggregate collector is
`scripts/car_model/ecsr_collect_phasek_barycentric_gate_summary.py`.

Outdoor-5 result:

| scene | selected | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report test dPSNR | report test dSSIM | report test dLPIPS | effective dPSNR | effective dSSIM | effective dLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | bary-delta v2wide s08 | yes | +0.000349 | -0.000044 | +0.000020 | +0.000872 | +0.000151 | -0.000389 | +0.000872 | +0.000151 | -0.000389 |
| flowers | Phase-J fallback | no | +0.000505 | -0.000076 | -0.000053 | -0.003515 | -0.000307 | +0.000180 | +0.000000 | +0.000000 | +0.000000 |
| garden | bary-delta v2wide s08 | yes | +0.000044 | -0.000035 | +0.000134 | +0.000669 | +0.000024 | -0.000033 | +0.000669 | +0.000024 | -0.000033 |
| stump | Phase-J fallback | no | +0.000597 | -0.000066 | -0.000207 | -0.000162 | +0.000001 | -0.000054 | +0.000000 | +0.000000 | +0.000000 |
| treehill | Phase-J fallback | no | -0.000019 | -0.000007 | +0.000012 | -0.000704 | -0.000005 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |

Mean effective outdoor-5 delta vs Phase-J:

| metric | mean delta |
|---|---:|
| PSNR | +0.000308 |
| SSIM | +0.000035 |
| LPIPS | -0.000084 |

Artifacts:

- aggregate:
  `outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/phasek_barycentric_gate_summary_outdoor5.md`;
- decisions:
  `outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/decisions/`;
- scene logs:
  `outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/*/phasek_barycentric_gate.log`.

This stronger validation confirms the gate is doing the right safety job: it
keeps `bicycle` and `garden`, where report-only test metrics are positive, and
rejects `flowers`, `stump`, and `treehill`, where the candidate is not a stable
improvement. It also confirms that the current representation update is still a
small effect-size method rather than a final top-conference-level breakthrough.

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
