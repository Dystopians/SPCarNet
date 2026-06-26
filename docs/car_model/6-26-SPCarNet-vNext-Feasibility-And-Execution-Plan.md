# SPCarNet vNext Feasibility and Execution Plan

Date: 2026-06-26

Scope: concise feasibility and execution plan for `docs/6-26-SPCarNet-vNext-ServerCodexPrompt.md`, based on the current SPCarNet status docs. This note records a plan only; it does not claim new experiment results.

## Verdict

The vNext direction is reasonable: Phase-J shows large train-certified residual gains, and v106 shows that a MeshSplatting-compatible surface residual representation is viable. The hard part is converting a render-time ELA/portfolio endpoint into a persistent face/UV/barycentric residual texture without train/test leakage or unsafe test-view regressions.

It is not reasonable as one more small v106 expert, alpha tweak, or gate variant. It should be treated as a staged representation project with a narrow pilot before full9. A pure fallback-to-v106 result is useful safety evidence, but it is not a successful vNext method.

## Current Facts to Respect

| line | status | vNext implication |
|---|---|---|
| Phase-J guarded ELA | 9/9 strict RGB wins vs clean; mean +1.3311 PSNR / +0.0347 SSIM / -0.0634 LPIPS; render-time, not baked | Use as teacher/upper bound, not as final representation claim |
| v104c | Stable full9 representation anchor, +0.6774 PSNR vs clean | Compare against it, but do not stop there |
| v106 POD-MoE base-preserve | Current verified quality line, +0.6796 PSNR vs clean; only about +0.0022 PSNR over v104c | vNext must improve candidate quality beyond v106 |
| v110/v110b | Train/even -> train/odd gates can still regress on held-out test vs v106 | Plain policy-val gates are insufficient |
| v113b/v113c | Safety repairs preserve or partially recover v106; not quality breakthroughs | Reuse lower-tail/OOT/fallback ideas |
| v114 | Active candidate-side attempt, not completed | Do not use as a claimed baseline until eval summaries exist |

## Scope

Minimum feasibility scope:

1. Build a train-only residual teacher cache on two scenes.
2. Fit a fixed-capacity face/UV residual texture from fit views only.
3. Select alpha, capacity, and fallback from policy-val only.
4. Evaluate test only after policy freeze.
5. Produce either a certified nonzero improvement beyond v106 on at least one scene, or a clear no-go diagnosis.

Minimum promotable milestone:

1. Full9 under one frozen train-only protocol.
2. 9/9 non-regressive or exact-tie vs the chosen parent on PSNR/SSIM/LPIPS.
3. At least 6/9 strict wins vs clean MeshSplatting and mean delta vs clean at least +0.5 PSNR with LPIPS improvement.
4. Nonzero certified vNext texture accepted on multiple scenes, with quantified v106-to-Phase-J gap closure.
5. Machine-readable provenance, policy decisions, fallback reasons, per-view metrics, size, runtime, triangle count, and residual texture budget.

## Main Risks

| risk | mitigation |
|---|---|
| Teacher cache or target surface maps accidentally read test GT | Require split/path/hash manifest plus `test_gt_used_for_selection=false`; fail closed if missing |
| Phase-J gains may be view-support effects, not persistent surface texture | First measure face/bin residual consistency before fitting capacity |
| Adaptive capacity overfits sparse residual-hot faces | Use fit/policy-val split, support counts, variance, positive-view fraction, CVaR, and OOT checks |
| Strict certificate collapses to no-op | Track accepted face/bin/pixel fraction; no promotion if fallback is the only passing behavior |
| Resource pressure blocks long jobs | Pilot first; compact caches; avoid dense duplicated tensors; account bytes before full9 |
| Baseline provenance is mixed or selected | Freeze parent, split, method names, renderer scaling, iteration, and artifact hashes before full9 |
| Compression claim drift | Report triangle count/storage/runtime, but do not market generic compression unless capacity reallocation actually changes budget |

## Staged Plan

### Stage 0 - Freeze Protocol

Write `outputs/carnet/spcarnet_vnext/cert_residual_surface_texture_<date>/protocol.json`.

Freeze pilot scenes `flowers` and `garden`; add `counter` before full9 because it exposed large-run pressure. Freeze parent rows, split rules, method names, renderer scaling, model roots, output root, and evaluator-only test-GT path.

Exit: each scene has a protocol JSON and required parent/teacher/source artifacts are present.

### Stage 1 - Build Residual Teacher Cache

Create or extend `scripts/car_model/ecsr_build_residual_teacher_cache_vnext.py`.

Reuse:

- `utils/evidence_lumigraph_adapter.py`
- `meshsplatopt_apply_evidence_lumigraph_adapter.py`
- `ecsr_apply_surface_residual_lumigraph_adapter.py`
- existing surface evidence cache scripts where compatible

Cache per train view: parent RGB, GT-parent residual, optional Phase-J/ELA teacher residual, face id, barycentric/UV bin, depth/boundary/normal cues where available, source paths, and hashes or timestamps.

Write:

- `teacher_cache/manifest.json`
- `teacher_cache/views/*.npz`
- `teacher_cache/no_test_gt_audit.json`
- `teacher_cache/residual_consistency_summary.json`

Exit: fit and policy-val caches load, and no test GT appears in candidate or policy inputs.

### Stage 2 - Fixed-Capacity Surface Texture

Create or extend `scripts/car_model/ecsr_apply_certified_residual_surface_texture_vnext.py`, starting from `ecsr_apply_surface_residual_region_texture_adapter.py`.

First candidate: fixed 8 or 16 bin texture per selected face/group, selected from fit residual support only. Render as:

```text
output = parent + confidence * residual
```

Write:

- `fields/<scene>_vnext_fixed_texture.pt`
- `reports/<scene>_fixed_texture_field_manifest.json`
- `reports/<scene>_fixed_texture_size_accounting.json`

Exit: alpha/confidence zero gives exact parent output; policy-val/test rendering does not read test GT for policy.

### Stage 3 - Train-Only Certificate

Add certificate logic to the vNext applicator or a separate gate. Reuse ideas from `meshsplatopt_v109_render_realized_parent_gate.py`, v113b lower-tail checks, and v113c frame-level OOT fallback.

Certificate fields:

- mean and lower-tail PSNR/SSIM/LPIPS gain vs parent;
- MSE direction and p95 MSE increase;
- positive-view fraction;
- support count and residual variance;
- OOT camera support;
- accepted face/bin/pixel fraction;
- fallback reason.

Write:

- `reports/<scene>_policy_decisions.json`
- `reports/<scene>_fallback_reasons.json`
- `reports/<scene>_certificate.md`

Exit: rejected candidates write exact parent outputs; accepted candidates have nonnegative tail evidence and no OOT breach.

### Stage 4 - Adaptive Capacity

Only after Stages 2 and 3 pass, add adaptive texture capacity.

Use capacity buckets such as `{4, 8, 16, 32}` per face/group. Allocate capacity to residual-hot, multiview-consistent, surface-addressable regions; downrank high variance, low support, OOT, and tail-risk regions.

Write:

- `reports/<scene>_capacity_plan.json`
- `reports/<scene>_capacity_policy_val_table.md`
- `reports/<scene>_storage_runtime_accounting.json`

Exit: adaptive capacity ties or beats fixed capacity on policy-val without worse tail risk; test is run only after capacity policy freezes.

### Stage 5 - Pilot Experiments

Run in this order:

1. `flowers`: strict safety/fallback case.
2. `garden`: OOT/frame-fallback case.
3. `counter`: add after storage path is stable.

Compare clean MeshSplatting, v104c, v106 parent, Phase-J teacher/upper bound, vNext fixed/no-adaptive, vNext adaptive/no-certificate, and full certified vNext.

Pilot promotion criteria:

- no test-GT selection audit failure;
- no regression vs v106 on accepted scenes, or exact parent fallback;
- at least one nonzero accepted vNext texture improves beyond v106;
- storage and runtime overhead are reported.

If the pilot only falls back to v106, stop before full9 and diagnose surface-addressability, residual consistency, capacity overfit, or OOT support.

### Stage 6 - Full9 Package

Add:

- `scripts/car_model/run_vnext_certified_residual_texture_full9.py`
- `scripts/car_model/assemble_vnext_certified_residual_texture_report.py`

Write:

- `docs/car_model/results/vnext_certified_residual_surface_texture_<date>/summary/full9_compare.md`
- `docs/car_model/results/vnext_certified_residual_surface_texture_<date>/summary/full9_compare.json`
- `docs/car_model/results/vnext_certified_residual_surface_texture_<date>/summary/per_view_tail_audit.json`
- `docs/car_model/assets/vnext_certified_residual_surface_texture_<date>/`

Full9 table must include clean, Phase-F compact parent if used, v104c, v106, Phase-J teacher, vNext fixed/no-adaptive, vNext no-certificate, and full vNext.

Exit: meet the minimum promotable milestone and issue one final recommendation: promote vNext, keep Phase-J as teacher/upper bound, or reject with diagnosed bottleneck.

## Go/No-Go

Proceed if disk/shared-memory pressure is handled, pilot artifacts exist, and the first deliverable is accepted as a leak-free pilot rather than a full9 headline.

Do not proceed directly to full9 if the cache cannot prove train-only provenance, fixed-capacity texture has no certified nonzero gain beyond v106, adaptive capacity adds overhead without policy-val tail benefit, or fallback is the only passing behavior.

## Recommended First Milestone

Build `vNext_certified_residual_surface_texture_pilot2` on `flowers` and `garden`.

Useful outcome:

1. certified nonzero residual surface texture improves at least one scene beyond v106 while the other is preserved by fallback; or
2. clean negative result identifying the blocker: teacher surface-addressability, capacity overfit, OOT support, or resource limits.

This is the smallest milestone that can decide whether the vNext prompt is a promising representation method rather than another safe gate around the current v106 parent.
