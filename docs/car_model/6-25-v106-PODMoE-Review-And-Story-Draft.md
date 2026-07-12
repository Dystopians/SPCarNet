# v106 POD-MoE Review and Paper-Story Draft

Date: 2026-06-25

Scope: review + paper-story synthesis for the v106 POD-MoE line. This is intentionally conservative. It records what can be claimed now, what remains diagnostic only, and how to write the paper story depending on whether the base-preserving boundary variant wins.

## 0. Honest Status

Current stable representation-field result:

- v104c is the current reliable fixed-policy representation-field anchor.
- v104c full9 is complete: `9 / 9` scenes are present and OK.
- v104c improves local clean MeshSplatting on full9 mean metrics: `25.829099 / 0.760727 / 0.268548` vs clean `25.151682 / 0.749018 / 0.287621`.
- v104c still trails the endpoint/reference line: `-0.652211 PSNR / -0.022949 SSIM / +0.044243 LPIPS`.

Current v106 status:

- v106 POD-MoE is the new research direction, not the current headline.
- It attempts to upgrade v104c from a single low-order surface residual function into a representation-level mixture/expert field.
- On `counter`, old/debtguard/cert POD-MoE variants slightly improve SSIM and LPIPS over v104c, but PSNR remains lower by about `0.011` to `0.017`.
- Render delta-MSE diagnostics are negative: for old, debtguard, and cert, only `4 / 30` counter views improve MSE versus v104c, while `26 / 30` worsen.
- The base-preserving boundary path is still under validation. It must not be described as solved or victorious until it passes a pre-declared promotion gate.

Safe one-sentence story:

> v104c is the current stable fixed-policy surface-field result; v106 POD-MoE is an early mixture/expert step toward closing the representation gap, with promising SSIM/LPIPS signals on counter but unresolved PSNR and MSE-direction regressions.

Unsafe one-sentence story:

> v106 POD-MoE is already the new best method.

## 1. Method Modules

### 1.1 Clean MeshSplatting

Clean MeshSplatting is the baseline:

```text
trained MeshSplatting checkpoint
-> direct held-out rendering
-> PSNR / SSIM / LPIPS
```

It does not explicitly model which surface regions have reliable multi-view residual evidence, where an occlusion boundary should trigger a fallback, or whether a correction is MSE-descent aligned.

### 1.2 v101/v102 Endpoint Reference

The endpoint/reference line is still the quality ceiling. It uses target-camera endpoint deltas and richer per-view evidence/fallback behavior. It is useful as a teacher/reference, but it is not a vanilla MeshSplatting checkpoint and not a train-only unseen-camera field.

Paper role:

- establishes the value of surface-addressed evidence repair;
- defines the quality gap that representation fields must close;
- should be shown as a reference row, not hidden.

### 1.3 v104c Shrink View-Affine Field

v104c stores a triangle-local, view-conditioned residual:

```text
[1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

Its key stabilization is fixed algebraic shrinkage: use view-affine residuals where support/rank/conditioning are reasonable, but shrink them toward a safer barycentric affine fallback. This makes v104c the current reliable fixed-policy field.

Paper role:

- main representation-field anchor;
- full9 evidence over clean MeshSplatting;
- honest bridge between endpoint repair and a compact surface field.

### 1.4 v106 POD-MoE Field

v106 POD-MoE uses:

```text
basis_type      = affine_barycentric_viewdir_pod_mixture
builder_variant = v106_perceptual_occlusion_detail_moe
field_variant   = pod_moe
base            = v104c-like shrink view-affine residual
experts         = detail, occlusion_boundary
```

Render-time intuition:

```text
base residual
+ detail expert weighted by detail cue and reliability
+ occlusion-boundary expert weighted by boundary cue and reliability
```

This is a real representation change. It is not merely another scalar threshold around v104c. The scientific target is to separate residual modes that v104c compresses into one low-order triangle-local function:

- normal supported residual behavior;
- high-frequency or perceptual detail residuals;
- occlusion-boundary residuals.

### 1.5 Debt Guard and MSE Certificate

The old POD-MoE output exposed a direction problem: many expert updates are not MSE-descent aligned relative to v104c.

Two repairs were then tested on counter:

- residual-debt guard: lowers expert reliability when expert coefficient shift is large relative to the base residual;
- weighted normal-equation lambda-star MSE certificate: stores expert MSE scale and damps experts when the fitted delta is not sufficiently MSE-aligned.

These are meaningful diagnostic mechanisms, but current evidence says they reduce the PSNR deficit rather than eliminate it.

### 1.6 Base-Preserving Boundary

The current code path supports `pod_base_keep_mode = base_preserving_boundary`. In this mode, boundary handling keeps the base residual instead of allowing the boundary cue to replace/suppress it. This is a plausible fix for over-aggressive occlusion-boundary behavior.

Claim boundary:

- It can be described as "under validation" or "a candidate boundary-safety modification."
- It cannot yet be described as "solved boundary handling" or "the final v106 result."

## 2. Quantitative Table Framework

### 2.1 Main Full9 Anchor: Clean vs v104c vs Endpoint

This table should appear early because it establishes the reliable story.

| method | scenes | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | local clean baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | current stable field |
| endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | quality ceiling |

| comparison | dPSNR | dSSIM | dLPIPS | interpretation |
|---|---:|---:|---:|---|
| v104c - clean | +0.677417 | +0.011709 | -0.019073 | reliable full9 field gain |
| v104c - endpoint/reference | -0.652211 | -0.022949 | +0.044243 | remaining endpoint-to-field gap |

### 2.2 Counter POD-MoE Diagnostic Table

This is the current v106 evidence table. It should be labeled counter-only.

| method | PSNR | SSIM | LPIPS | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c | status |
|---|---:|---:|---:|---:|---:|---:|---|
| clean MeshSplatting | 26.751774 | 0.862055 | 0.252003 | - | - | - | baseline |
| v104c shrink view-affine | 27.498068 | 0.867420 | 0.238986 | 0.000000 | 0.000000 | 0.000000 | stable anchor |
| v106 POD-MoE old | 27.480730 | 0.867727 | 0.238923 | -0.017338 | +0.000308 | -0.000064 | SSIM/LPIPS up, PSNR down |
| v106 POD-MoE debtguard | 27.486620 | 0.867725 | 0.238849 | -0.011448 | +0.000305 | -0.000137 | smaller PSNR deficit |
| v106 POD-MoE cert | 27.486565 | 0.867725 | 0.238849 | -0.011503 | +0.000305 | -0.000137 | similar to debtguard |
| endpoint/reference | 28.442907 | 0.893696 | 0.186557 | +0.944839 | +0.026276 | -0.052429 | quality ceiling |

Interpretation:

- All POD-MoE variants beat clean on counter.
- None strictly beats v104c because PSNR regresses.
- SSIM/LPIPS gains are small enough that qualitative/error-map verification is mandatory.

### 2.3 MSE Direction Diagnostic Table

This table is important because it prevents overclaiming metric cherry-picks.

| candidate vs v104c | views | MSE-improved | MSE-worse | mean delta MSE | mean 2ed | mean d2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| old POD-MoE | 30 | 4 | 26 | +0.00000711 | +0.00000342 | 0.00000369 | mostly MSE-worse |
| debtguard POD-MoE | 30 | 4 | 26 | +0.00000479 | +0.00000283 | 0.00000196 | improved damping, still worse |
| cert POD-MoE | 30 | 4 | 26 | +0.00000481 | +0.00000284 | 0.00000197 | similar to debtguard |

Interpretation:

- Debtguard/cert reduce mean delta-MSE magnitude relative to old POD-MoE.
- They do not change the sign or view-count diagnosis.
- This explains the mixed metric behavior: perceptual/structure metrics can move slightly while PSNR/MSE worsens.

### 2.4 Required Promotion Tables

Before v106 can replace v104c in the story, add these tables.

Hard-triad gate:

| scene | v104c PSNR | v106 PSNR | dPSNR | v104c SSIM | v106 SSIM | dSSIM | v104c LPIPS | v106 LPIPS | dLPIPS | MSE worse views | pass? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| counter | filled | filled | filled | filled | filled | filled | filled | filled | filled | filled | yes/no |
| kitchen | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| bonsai | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Full9 fixed-policy gate:

| method | scenes OK | mean PSNR | mean SSIM | mean LPIPS | strict scene wins vs v104c | strict scene wins vs clean | endpoint gap PSNR | endpoint gap LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v104c | 9 | 25.829099 | 0.760727 | 0.268548 | anchor | 9/9 vs clean | -0.652211 | +0.044243 |
| v106 base-preserve candidate | pending | pending | pending | pending | pending | pending | pending | pending |

## 3. Qualitative Display Plan

Qualitative evidence should answer one question:

> Are the tiny SSIM/LPIPS gains from POD-MoE real local improvements in detail or occlusion regions, or metric noise paired with MSE-positive drift?

Recommended panel layout:

```text
GT | clean | v104c | v106 POD-MoE | endpoint/reference | error maps
```

For counter, include at least three view buckets:

- best MSE views where POD-MoE improves over v104c, e.g. `00000.png`, `00008.png`, `00029.png`;
- worst MSE views where POD-MoE hurts, e.g. `00009.png`, `00014.png`, `00016.png`, `00017.png`, `00010.png`;
- SSIM/LPIPS-favorable crops if they are visually interpretable.

Recommended crop policy:

- use absolute-error maps and delta-error maps, not only RGB crops;
- show the same crop for v104c, v106, endpoint, and GT;
- annotate scene/view/crop coordinates and local metric deltas;
- avoid hand-picked full-frame images where differences are invisible;
- separate endpoint-quality panels from v106 representation-field panels.

Expected qualitative outcomes:

- If improvements are concentrated on texture/detail or boundary structure, the POD-MoE story is meaningful even before full9.
- If improvements are not visually localizable, treat the SSIM/LPIPS gain as too small to drive the paper story.

## 4. Review Checklist

A reviewer will likely ask the following. The paper/story must answer them directly.

| question | required answer |
|---|---|
| Does v106 beat v104c? | Not yet. Counter SSIM/LPIPS improve slightly, but PSNR and MSE direction remain worse. |
| Is v104c still relevant? | Yes. It is the current full9 fixed-policy representation-field anchor. |
| Is v106 a representation-level change? | Yes. It adds a POD-MoE field with v104c-like base plus detail and occlusion-boundary experts. |
| Is the method train-only unseen-camera generalization? | No. Current v104c/v106 field line uses v102 target-camera endpoint deltas as teacher. No held-out GT is used for policy, but this is still target-camera endpoint distillation. |
| Is base-preserving boundary solved? | No. It is under validation. |
| Why care about v106 if PSNR drops? | Because it tests the right capacity hypothesis: one low-order residual is insufficient, and separate detail/boundary experts may target the endpoint gap. But it remains diagnostic until gates pass. |

## 5. Must-Admit Shortcomings

These limitations should be explicit, not buried.

1. v104c is the current stable field result, not v106.
2. v106 evidence is counter-only so far.
3. v106 old/debtguard/cert do not strictly beat v104c because PSNR regresses.
4. MSE diagnostics are unfavorable: `26 / 30` counter views are worse for all three POD-MoE variants.
5. SSIM/LPIPS gains are tiny and require crop/error-map verification.
6. The endpoint/reference gap remains large, especially LPIPS/detail.
7. Current fields use v102 target-camera endpoint deltas as teacher; this is not a fully train-only unseen-camera representation claim.
8. Base-preserving boundary is still a candidate fix, not an established win.
9. Current artifacts for POD-MoE are partly under `/dev/shm`; before paper/package use, durable summaries and manifests should be copied or regenerated under `outputs/`.

## 6. How to Write It If Base-Preserve Wins

Use this only if base-preserving boundary passes a pre-declared gate against v104c.

Minimum gate:

- counter improves or ties PSNR while retaining SSIM/LPIPS gains;
- MSE diagnostic no longer shows majority-view worsening, or the worse views have a clear qualitative/perceptual justification;
- hard-triad fixed policy passes on `counter/kitchen/bonsai`;
- full9 expansion is complete before any headline claim.

If it wins on counter only:

> The base-preserving boundary variant fixes the first POD-MoE failure mode on counter: it retains the v104c base residual near occlusion boundaries while letting detail/boundary experts contribute only as additive evidence. This converts the earlier SSIM/LPIPS-only signal into a cleaner candidate for hard-triad validation. We do not yet claim full9 superiority.

If it wins on hard triad:

> Base-preserving POD-MoE is the first mixture/expert surface-field candidate to improve the hard-triad anchor over v104c under a fixed policy. This supports the paper hypothesis that the endpoint-to-field gap is partly a residual-mode separation problem, not only a shrinkage/calibration problem. Full9 validation remains required for the main claim.

If it wins on full9:

> v106 POD-MoE upgrades v104c from a single shrink view-affine residual into a base-preserving mixture of detail and occlusion-boundary experts. Under the same local full9 protocol, it improves over the v104c fixed-policy field while preserving the clean MeshSplatting gains. This makes representation-level residual-mode separation the new main method story.

Even in the winning case, keep the endpoint/reference row unless v106 also closes it.

## 7. How to Write It If Base-Preserve Does Not Win

If base-preserve does not beat v104c, do not bury v106. Use it as a negative-but-useful diagnostic.

Counter-only failure wording:

> POD-MoE tests the right representation hypothesis, but the current boundary/detail experts are not yet MSE-aligned with the v104c base. Debtguard and MSE certification reduce the size of the harmful delta, yet most counter views still worsen in MSE and PSNR remains below v104c. We therefore keep v104c as the stable field result and treat v106 as a diagnostic for the next representation design.

Hard-triad failure wording:

> The counter SSIM/LPIPS gains do not generalize cleanly to the hard triad. This suggests that expert assignment and boundary/detail cues remain scene-specific. The paper should present POD-MoE as an ablation that motivates stronger evidence-calibrated experts, not as the main method.

Full9 failure wording:

> v106 POD-MoE does not replace v104c under fixed full9 evaluation. The result is still informative: it shows that naive expert capacity can improve perceptual metrics locally but may introduce MSE-positive residual drift. The reliable paper claim remains v104c over clean MeshSplatting, with POD-MoE as future work toward closing the endpoint gap.

## 8. Suggested Paper Framing

Recommended structure:

1. Start with the problem: clean MeshSplatting lacks explicit surface-evidence repair.
2. Show v101/v102 endpoint/reference as the quality ceiling and teacher.
3. Present v104c as the stable fixed-policy representation-field result that beats clean full9.
4. Introduce v106 POD-MoE as the next representation hypothesis: residual-mode separation via detail and boundary experts.
5. Report counter results honestly: SSIM/LPIPS small gains, PSNR/MSE still negative.
6. Use base-preserve validation as the decision point for whether v106 becomes a method result or a diagnostic ablation.

Safe abstract-style paragraph:

> We first establish a stable fixed-policy surface residual field, v104c, that improves local clean MeshSplatting across full9 scenes while remaining below the endpoint/reference quality ceiling. We then investigate v106 POD-MoE, a mixture/expert field that separates a v104c-like base residual from detail and occlusion-boundary experts. On counter, POD-MoE variants slightly improve SSIM and LPIPS over v104c but still regress PSNR and worsen MSE for most views. These results motivate base-preserving boundary validation and show that representation-level residual-mode separation is promising but not yet closed.

## 9. Evidence Paths

Stable v104c evidence:

```text
docs/car_model/6-25-v104c-Full9-PaperLoop-Technical-Report.md
docs/car_model/6-25-v104c-Subagent-Synthesis.zh.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.json
```

v106 POD-MoE counter evidence:

```text
/dev/shm/peilincai_spcarnet_v106_podmoe_counterfirst_20260625_reports/v106_podmoe_counter_summary_v2.md
/dev/shm/peilincai_spcarnet_v106_podmoe_debtguard_counter_20260625_reports/v106_podmoe_debtguard_counter_summary.md
/dev/shm/peilincai_spcarnet_v106_podmoe_cert_counter_20260625_reports/v106_podmoe_cert_counter_summary.md
```

MSE diagnostics:

```text
/dev/shm/peilincai_spcarnet_v106_podmoe_counterfirst_20260625_reports/v106_podmoe_vs_v104c_delta_mse.md
/dev/shm/peilincai_spcarnet_v106_podmoe_debtguard_counter_20260625_reports/v106_podmoe_debtguard_vs_v104c_delta_mse.md
/dev/shm/peilincai_spcarnet_v106_podmoe_cert_counter_20260625_reports/v106_podmoe_cert_vs_v104c_delta_mse.md
```

Implementation identity:

```text
render.py
scripts/car_model/build_v105_evidence_gated_mixture_field.py
scripts/car_model/run_v105_evidence_gated_mixture_scene.py
scripts/car_model/summarize_v105_evidence_gated_mixture.py
```

