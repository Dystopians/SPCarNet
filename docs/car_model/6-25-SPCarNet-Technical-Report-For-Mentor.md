# SPCarNet Technical Report for Mentor PPT

Date: 2026-06-25  
Scope: review + paper-story synthesis for mentor slides. This draft is intentionally conservative: it summarizes the current evidence and open gaps, and does not claim the method is already a complete top-conference solution.

## 0. Slide-Level Takeaway

The clean MeshSplatting baseline is:

```text
train MeshSplatting checkpoint
-> directly render held-out views
-> evaluate PSNR / SSIM / LPIPS
```

SPCarNet adds an evidence layer around the trained MeshSplatting surface:

```text
MeshSplatting surface as address space
-> collect train/support-view evidence
-> estimate reliable residual repair on visible triangles
-> shrink, gate, or fall back when evidence is weak
-> render and evaluate under the same held-out protocol
```

The most honest current story for tomorrow's mentor PPT:

- **Quality ceiling:** the endpoint/reference line is still best, with full9 mean `26.481310 / 0.783675 / 0.224305`.
- **Current stable representation field:** v104c shrink view-affine field is the best stable method to present as a representation-side result. It beats the local clean MeshSplatting full9 baseline on mean metrics: `25.829099 / 0.760727 / 0.268548` vs clean `25.151682 / 0.749018 / 0.287621`.
- **New v106 POD-MoE evidence:** base-preserving boundary is the first POD-MoE counter variant that strictly beats v104c on PSNR, SSIM, and LPIPS, and its MSE-direction diagnostic is much healthier. The margin is still tiny and counter-only, so this is a milestone probe, not the new full9 headline yet.
- **Main remaining gap:** v104c/v106 still compress a strong per-view endpoint into a low-order surface field. The endpoint gap, especially LPIPS/detail, is still visible and should be shown.

Recommended slide sentence:

> SPCarNet shows that MeshSplatting can be improved by surface-addressed evidence repair; v104c is the current stable full9 representation-field result, while v106 POD-MoE base-preserve is the first expert-field variant that beats v104c on the counter/kitchen/bonsai hard-triad and now needs full9 validation.

## 1. Method Modules in Plain Language

### 1.1 MeshSplatting Baseline

MeshSplatting trains a surface-aware scene representation and then directly renders held-out views. The baseline does not explicitly ask whether a triangle has enough multi-view evidence, whether a residual repeats across views, or whether a correction should be rejected near occlusion boundaries.

That makes it a strong and fair baseline, but also leaves a natural repair opportunity: local texture/color/detail errors that are consistently visible in train/support views can be measured and reused.

### 1.2 Surface Evidence

SPCarNet treats the MeshSplatting surface as an address system. For each visible triangle/face region, it records evidence such as:

- which train/support views observed the surface;
- how consistent the residual signal is;
- whether the local region is sparse, unstable, or near an occlusion/detail boundary;
- whether a correction has enough support to be trusted.

The key idea is not "add a postprocess filter." The repair is tied to 3D surface visibility and support.

### 1.3 Endpoint / Reference Line

The endpoint/reference line, currently represented by the v101/v102-style evidence endpoint, is the strongest image-quality result. It can use richer per-view support and fallback behavior than a compact field.

Its role in the paper story:

- demonstrates the quality ceiling of evidence-based repair;
- provides a teacher/reference for field distillation;
- should not be described as a vanilla MeshSplatting checkpoint or a completed train-only unseen-camera field.

### 1.4 v104c Shrink View-Affine Field

v104c is the current stable representation-field method. It stores a triangle-local, view-conditioned residual:

```text
[1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z]
  -> RGB residual
```

The important stabilization is shrinkage. Raw view-affine fitting can over-trust poorly supported triangles. v104c estimates a view-conditioned field, then shrinks unreliable coefficients toward a safer barycentric affine fallback. In plain terms:

> use view direction where evidence supports it, but damp it instead of letting weak triangles make aggressive corrections.

This is why v104c is currently the safest representation-side headline.

### 1.5 v106 POD-MoE Counter Probe

v106 POD-MoE extends the v104c-like field with two extra residual experts:

```text
base: v104c-like shrink view-affine residual
expert 1: detail residual
expert 2: occlusion-boundary residual
```

The local manifest names this as:

```text
basis_type      = affine_barycentric_viewdir_pod_mixture
builder_variant = v106_perceptual_occlusion_detail_moe
field_variant   = pod_moe
expert_names    = detail, occlusion_boundary
```

Intuition:

- v104c uses one low-order residual function per triangle;
- POD-MoE tries to separate "normal supported residual", "perceptual/detail residual", and "occlusion-boundary residual";
- the hope is to recover some endpoint detail capacity without returning to a full endpoint sidecar.

Current status: the base-preserving boundary variant now strictly beats v104c on counter across PSNR, SSIM, and LPIPS. The margin is small, so this should be presented as a promising mechanism milestone rather than as a final full9 replacement.

## 2. Difference from MeshSplatting Baseline

| Aspect | Clean MeshSplatting | SPCarNet current line |
|---|---|---|
| Render source | Direct checkpoint render | Checkpoint render plus surface-addressed residual field/endpoint |
| Evidence awareness | No explicit train/support-view trust layer | Uses visible-surface evidence, support, gates, shrink, fallback |
| Local correction | Whatever the checkpoint learned | Adds residual correction only where evidence supports it |
| Safety behavior | No explicit correction rejection | Weak regions can shrink or fall back |
| Current best quality | Local clean baseline | Endpoint/reference is best; v104c is stable field version |
| Paper risk | Strong standard baseline | Must avoid overstating endpoint sidecar as vanilla checkpoint or train-only generalization |

For mentor slides, the clean comparison is:

> MeshSplatting gives us the trained surface representation. SPCarNet asks which parts of that surface can be safely repaired using multi-view evidence, and how much of the endpoint repair can be baked back into a compact field.

## 3. Current Evidence

### 3.1 Full9 Mean: Clean vs v104c vs Endpoint

Full9 local protocol, same scene set and evaluator.

| Method | PSNR | SSIM | LPIPS | Interpretation |
|---|---:|---:|---:|---|
| Clean MeshSplatting baseline | 25.151682 | 0.749018 | 0.287621 | Local clean baseline |
| v104c shrink view-affine field | 25.829099 | 0.760727 | 0.268548 | Current stable representation field |
| Endpoint/reference | 26.481310 | 0.783675 | 0.224305 | Current quality ceiling |

| Comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104c - clean | +0.677417 | +0.011709 | -0.019073 |
| v104c - endpoint/reference | -0.652211 | -0.022948 | +0.044243 |

Takeaway:

- v104c is real evidence over clean on full9, not just a counter-only probe.
- v104c still has a meaningful endpoint gap, so the paper story should include both rows.

### 3.2 Counter: Clean vs v104c vs v106 POD-MoE vs Endpoint

| Method | PSNR | SSIM | LPIPS | Status |
|---|---:|---:|---:|---|
| Clean MeshSplatting | 26.751774 | 0.862055 | 0.252003 | Baseline |
| v104c shrink view-affine | 27.498068 | 0.867420 | 0.238986 | Current stable field anchor |
| v106 POD-MoE old | 27.480730 | 0.867727 | 0.238923 | Counter-only diagnostic; PSNR below v104c |
| v106 POD-MoE base-preserve | 27.499645 | 0.867521 | 0.238847 | Counter-only milestone; all three metrics beat v104c |
| Endpoint/reference | 28.442907 | 0.893696 | 0.186557 | Quality ceiling |

| Comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v106 POD-MoE old - clean | +0.728956 | +0.005672 | -0.013081 |
| v106 POD-MoE old - v104c | -0.017338 | +0.000308 | -0.000064 |
| v106 POD-MoE base-preserve - clean | +0.747871 | +0.005466 | -0.013156 |
| v106 POD-MoE base-preserve - v104c | +0.001577 | +0.000102 | -0.000139 |
| v106 POD-MoE base-preserve - endpoint/reference | -0.943262 | -0.026175 | +0.052290 |

Interpretation:

- The old POD-MoE was not a strict upgrade because PSNR dropped.
- The base-preserving boundary variant fixes the counter PSNR regression and keeps SSIM/LPIPS above v104c.
- The gains are directionally useful but very small.
- The endpoint gap remains large, so v106 should be framed as "capacity/structure milestone under validation" rather than a completed method.

### 3.3 Evidence Hygiene

Useful current paths:

```text
docs/car_model/6-25-v104c-Full9-PaperLoop-Technical-Report.md
docs/car_model/6-25-v104c-Subagent-Synthesis.zh.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.json
/dev/shm/peilincai_spcarnet_v106_podmoe_counterfirst_20260625_reports/counter/counter_v105_evidence_gated_mixture_report.json
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/counter/counter_v105_evidence_gated_mixture_report.json
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/v106_podmoe_basepreserve_counter_summary.json
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/v106_podmoe_basepreserve_vs_v104c_delta_mse.json
```

Important artifact note:

- `outputs/carnet/meshsplatopt/ecsr_phase_v106_optimal_risk_counter_probe_20260625/` is a different v106 optimal-risk probe, not the POD-MoE number above.
- The POD-MoE evidence currently lives under `/dev/shm/...podmoe...`; before final paper/package work, it should be materialized into a durable `outputs/` summary with exact manifest and report paths.

## 4. What Is Not Finished Yet

### 4.1 Representation Gap

The strongest endpoint can make richer per-view decisions. v104c and v106 compress that behavior into a compact surface field. This compression loses:

- per-view support/fallback decisions;
- occlusion-context decisions;
- high-frequency detail;
- possibly perceptual structure that LPIPS is sensitive to.

v106 targets this gap with detail and occlusion-boundary experts. The base-preserving boundary repair makes the counter result directionally positive, but the endpoint gap remains large.

### 4.2 Generalization Boundary

The current v104c/v106 field line is distilled from v102 target-camera endpoint deltas. The reports state no held-out target GT is used for policy selection, but this is still not the same as a fully train-only unseen-camera representation.

Safe wording:

> Current field experiments are target-camera endpoint distillation probes with strict held-out metric reporting.

Unsafe wording:

> The current field is already a vanilla MeshSplatting checkpoint or fully solved unseen-view generalization.

### 4.3 v106 Evidence Is Single-Scene

v106 POD-MoE base-preserve has a hard-triad win on `counter/kitchen/bonsai`. It should not be promoted to full paper headline until it passes:

- counter strict gate against v104c on all three metrics; this is now passed for base-preserve;
- hard-triad expansion on `counter/kitchen/bonsai`; this is now passed for base-preserve;
- full9 fixed-policy run;
- visual/error-map inspection showing the SSIM/LPIPS gain is real and not metric noise.

### 4.4 Artifact and Reporting Cleanup

The v106 POD-MoE report file still uses the generic report title `v105 Evidence-Gated Mixture Report`. The manifest identity is correct, but for mentor/PPT and later paper tracking this can confuse readers. A durable summary should rename the row as v106 POD-MoE and include:

- field variant and builder identity;
- exact metrics and deltas;
- no-test-GT policy statement;
- endpoint-distillation boundary;
- comparison against v104c and endpoint/reference.

## 5. Next Experimental Closure Loop

### 5.0 New v106 Diagnostic Added on 2026-06-25

After the first v106 POD-MoE counter probe, we added a render-level MSE direction diagnostic:

```text
scripts/car_model/diagnose_render_delta_mse.py
```

It compares a candidate render against a base render and decomposes the candidate's MSE change:

```text
delta_mse = 2 * (base - gt) * (candidate - base) + (candidate - base)^2
```

For the old `v106_podmoe_counter` against v104c on `counter`, the diagnostic found:

| item | value |
|---|---:|
| views | 30 |
| MSE-improved views | 4 |
| MSE-worse views | 26 |
| mean delta MSE | +0.00000711 |
| mean 2ed cross term | +0.00000342 |
| mean d2 energy | 0.00000369 |

Artifact paths:

```text
/dev/shm/peilincai_spcarnet_v106_podmoe_counterfirst_20260625_reports/v106_podmoe_vs_v104c_delta_mse.json
/dev/shm/peilincai_spcarnet_v106_podmoe_counterfirst_20260625_reports/v106_podmoe_vs_v104c_delta_mse.md
```

Interpretation: the old POD-MoE expert output is not merely too strong; in most views it points in an MSE-positive direction relative to v104c. This explains why SSIM/LPIPS can move slightly up while PSNR drops.

Three follow-up repairs were implemented for validation:

1. **POD-MoE residual-debt guard:** reduces expert reliability when the expert coefficient shift from the v104c-like base is large relative to the base residual.
2. **POD-MoE MSE-direction certificate:** stores a closed-form per-triangle/per-expert `lambda*` from the weighted normal equations and applies `triangle_expert_mse_scale` at render time.
3. **POD-MoE base-preserving boundary:** keeps the v104c-like base residual intact at boundary pixels instead of letting the boundary expert suppress or replace it.

The MSE-direction certificate is a method change, not a scene parameter scan. It asks: "if this expert delta is applied to the base residual under its own support stratum, does it move along the MSE descent direction, and how far should it be trusted?"

Counter validation summary:

```text
debtguard counter:
  field_root  = /dev/shm/peilincai_spcarnet_v106_podmoe_debtguard_counter_20260625_field
  report_root = /dev/shm/peilincai_spcarnet_v106_podmoe_debtguard_counter_20260625_reports
  output      = ours_26000_v106_podmoe_debtguard_counter

certificate counter:
  field_root  = /dev/shm/peilincai_spcarnet_v106_podmoe_cert_counter_20260625_field
  report_root = /dev/shm/peilincai_spcarnet_v106_podmoe_cert_counter_20260625_reports
  output      = ours_26000_v106_podmoe_cert_counter

base-preserve counter:
  field_root  = /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_field
  report_root = /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports
  output      = ours_26000_v106_podmoe_basepreserve_counter
```

MSE-direction comparison against v104c on counter:

| candidate | MSE-improved views | MSE-worse views | mean delta MSE | mean 2ed | mean d2 |
|---|---:|---:|---:|---:|---:|
| old POD-MoE | 4 / 30 | 26 / 30 | +0.00000711 | +0.00000342 | 0.00000369 |
| debtguard POD-MoE | 4 / 30 | 26 / 30 | +0.00000479 | +0.00000283 | 0.00000196 |
| cert POD-MoE | 4 / 30 | 26 / 30 | +0.00000481 | +0.00000284 | 0.00000197 |
| base-preserve POD-MoE | 23 / 30 | 7 / 30 | -0.00000026 | -0.00000068 | 0.00000042 |

Interpretation: base-preserving boundary is not just a tiny metric change. It flips the render-level MSE direction from mostly worse to mostly improved on counter. This is the strongest evidence so far that the POD-MoE failure mode was boundary/base suppression rather than simply too much expert capacity.

Hard-triad validation:

```text
hard-triad kitchen:
  field_root  = /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_field
  report_root = /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports
  output      = ours_26000_v106_podmoe_basepreserve_kitchen

hard-triad bonsai:
  field_root  = /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_field
  report_root = /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports
  output      = ours_26000_v106_podmoe_basepreserve_bonsai
```

Hard-triad metric deltas against v104c:

| scene | dPSNR | dSSIM | dLPIPS | MSE-improved views | MSE-worse views | mean delta MSE |
|---|---:|---:|---:|---:|---:|---:|
| counter | +0.001577 | +0.000102 | -0.000139 | 23 / 30 | 7 / 30 | -0.00000026 |
| kitchen | +0.001595 | +0.000062 | -0.000206 | 30 / 35 | 5 / 35 | -0.00000050 |
| bonsai | +0.005213 | +0.000154 | -0.000136 | 36 / 37 | 1 / 37 | -0.00000017 |
| mean | +0.002795 | +0.000106 | -0.000160 | 89 / 102 | 13 / 102 | - |

Interpretation: base-preserve passes the hard-triad fixed-policy gate. The metric gains are small, but their signs are consistent and the render-level MSE direction is mostly improved.

Promotion rule after hard-triad:

- must pass field/render/eval identity checks;
- must beat clean MeshSplatting clearly;
- should not regress v104c PSNR;
- SSIM/LPIPS gains over v104c must be retained or explained with qualitative evidence;
- only then expand to full9.

### Step 1: Lock the Current v104c Story

Use v104c as the stable representation-field slide:

- full9 mean over clean;
- 9-scene evidence from the v104c summary;
- endpoint/reference gap shown explicitly;
- no claim that v104c reaches endpoint quality.

Deliverable for PPT: one table plus one simple module diagram.

### Step 2: Turn v106 POD-MoE Into a Clean Counter Diagnostic

Before expanding, make the counter result self-contained:

- move or summarize the `/dev/shm` POD-MoE report into durable `outputs/` or docs evidence;
- fix report naming from v105 generic title to v106 POD-MoE;
- add a tiny comparison table: v104c, old POD-MoE, debtguard/cert POD-MoE, base-preserve POD-MoE;
- inspect one or two counter crops/error maps where base-preserve improves MSE and where it still hurts.

Decision rule:

> If base-preserve's gains are visible and localized to detail/occlusion regions, expand to hard-triad/full9. If they are not visible, treat the metric gains as too small for the paper story and move to a stronger MSE-descent projection.

### Step 3: Full9 Expansion After the Hard-Triad Gate

The hard-triad gate is now passed. The next loop is:

1. Run the remaining six full9 scenes with the same fixed POD-MoE base-preserve policy.
2. Summarize all nine scenes against clean, v104c, and endpoint/reference.
3. Inspect worst-view error maps for scenes that regress or show tiny gains.
4. If full9 remains positive, promote v106 base-preserve as the current active expert-field method.

### Step 4: Close the Endpoint-to-Field Gap Scientifically

Use endpoint/reference as a teacher, but diagnose the gap rather than hiding it:

- per-view LPIPS/error-map gap between endpoint and v104c/v106;
- classify gap regions as detail, occlusion boundary, sparse support, view-dependent color, or fallback failure;
- map those regions to expert design in POD-MoE;
- test whether each expert actually improves its intended region.

### Step 5: Paper Claim Boundary

Current safe paper-level claim:

> Evidence-certified surface repair improves local MeshSplatting baselines; v104c shows a fixed view-conditioned surface field can recover part of the endpoint gain on full9, while v106 POD-MoE base-preserve is the first expert-field variant that beats v104c on the counter/kitchen/bonsai hard-triad and is now ready for full9 validation.

Current unsafe claims:

- "SPCarNet is fully solved."
- "v106 is the new best full9 method."
- "The current field matches endpoint/reference."
- "The current field is a vanilla MeshSplatting checkpoint."
- "The current target-camera distillation proves train-only unseen-camera generalization."

## 6. Suggested PPT Outline

1. **Problem:** MeshSplatting is strong, but direct checkpoint rendering has local repeatable errors.
2. **Idea:** use the mesh surface as an evidence address space for residual repair.
3. **System:** evidence collection, residual field/endpoint, gate/shrink/fallback, held-out evaluation.
4. **Main evidence:** v104c full9 beats clean, endpoint remains quality ceiling.
5. **New probe:** v106 POD-MoE adds detail and occlusion-boundary experts; base-preserving boundary fixes the counter PSNR regression and makes the MSE direction mostly positive.
6. **Honest gap:** representation field still trails endpoint, especially perceptual/detail quality.
7. **Next loop:** durable v106 report, crop/error-map diagnosis, remaining full9 scenes, then promote only if full9 stays positive.

## 7. One-Minute Verbal Script

MeshSplatting gives us a trained surface representation. SPCarNet does not throw that away; it adds an evidence layer on top. We use train/support views to ask which surface regions have reliable residual signals, then transfer only the supported corrections to held-out views, with shrink or fallback where evidence is weak.

The strongest endpoint/reference still has the best quality, but it is not yet the representation we ultimately want. v104c is the current stable attempt to bake part of that endpoint behavior into a compact view-conditioned surface field, and it improves the local full9 clean baseline. The new v106 POD-MoE probe is a capacity step: it adds detail and occlusion-boundary experts. The first versions improved SSIM/LPIPS but lost PSNR; the base-preserving boundary version now beats v104c on counter, kitchen, and bonsai across all three metrics and flips the MSE-direction diagnostic from mostly worse to mostly improved. That makes it a real milestone, but still not a full9 headline until the remaining six scenes finish.

The next research loop is to run the remaining full9 scenes, diagnose where endpoint still beats the field, and use error maps to decide whether v106 base-preserve is a promotable active method or whether we need the stronger v107 MSE-descent projection.
