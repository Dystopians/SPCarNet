# SPCarNet / MeshSplatOpt Current Method Technical Report

Date: 2026-06-22

Audience: mentor update / PPT preparation

Status: current accepted endpoint is **Phase-J compact MeshSplatting + train-only Evidence Lumigraph Adapter**. Newer v25/v26 repair attempts are not yet accepted as the headline method.

---

## 1. Executive Summary

We start from the original MeshSplatting representation and ask a focused question:

> Can we keep the engine-friendly opaque triangle mesh property of MeshSplatting, while improving held-out rendering quality and reducing unnecessary topology?

The current strongest answer is **yes, on the validated Mip-NeRF360 full9 evidence set**.

Compared with our locally reproduced clean MeshSplatting baseline selected from clean `26000` and `30000` checkpoints, Phase-J achieves:

| comparison | scenes | dPSNR | dSSIM | dLPIPS | mean triangle reduction |
|---|---:|---:|---:|---:|---:|
| Phase-J vs local clean MeshSplatting | `9 / 9` strict RGB wins | `+1.3311` | `+0.0347` | `-0.0634` | `7.6479%` |

Compared with the MeshSplatting paper table on Mip-NeRF360, after normalizing the metric order to `PSNR / SSIM / LPIPS`, Phase-J is also above the reported average:

| row | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` |
| our local clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` |
| Phase-J current method | `26.4828` | `0.7837` | `0.2243` |
| Phase-J minus paper table | `+1.7017` | `+0.0555` | `-0.0865` |

Important qualification:

- The local clean MeshSplatting baseline is **our own reproduced baseline**, not copied from the paper.
- The paper-table comparison is useful as an external sanity check, but the fair primary claim should use local same-code, same-data, same-metrics comparison.
- The newest v25 witness-CVaR branch did **not** improve over Phase-J. It is a useful negative diagnosis, not the current headline method.

Primary evidence:

- Current Phase-J full9 summary: [`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`](../../outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md)
- Current README result block: [`README.md`](../../README.md)
- v25 negative result: [`docs/car_model/5-30-WitnessGroupCVaR-v25-Log.md`](5-30-WitnessGroupCVaR-v25-Log.md)
- MeshSplatting paper table source: <https://arxiv.org/html/2512.06818v1>

---

## 2. Problem Setting

MeshSplatting is already a strong baseline because it directly produces colored opaque meshes that can be rendered efficiently and are more compatible with graphics/AR/VR pipelines than point-based Gaussian representations.

However, the clean checkpoints still show three practical issues in our local runs:

1. **Local residual texture errors**: fine foliage, bark, bench slats, and indoor high-frequency patterns can appear smoothed or locally color-biased.
2. **Checkpoint overfitting sensitivity**: clean `30000` is often worse than clean `26000` under held-out scoring, so "train longer" is not a reliable answer.
3. **Topology redundancy**: not all triangles are equally useful for held-out rendering. Some can be safely removed if protected by a render/geometry gate.

Our method is designed around a conservative principle:

> Do not blindly rewrite the mesh. Compress only when train-view evidence says the surface is safe, and repair appearance only when train-only residual evidence certifies the correction.

---

## 3. Baseline Protocol

### 3.1 Local clean MeshSplatting baseline

For every Mip-NeRF360 scene, we compare against local clean MeshSplatting checkpoints and select the clean baseline by held-out test score:

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

The current README records that clean `26000` is selected over clean `30000` on all nine scenes under this held-out score. This is important because train metrics or training length would bias the baseline toward longer but sometimes overfit checkpoints.

### 3.2 Paper-table baseline

The MeshSplatting paper reports Mip-NeRF360 average:

```text
PSNR 24.78 / SSIM 0.728 / LPIPS 0.310
```

The paper-table comparison is not the strictest fairness baseline because it is an external number, but it verifies that our local clean reproduction is not artificially weak and that Phase-J remains above the paper-level MeshSplatting average.

---

## 4. Method Overview

The current accepted method can be described as:

> **A train-only evidence-certified compact-and-repair wrapper around MeshSplatting.**

It has three load-bearing modules.

### 4.1 Sparse-occlusion protected compaction

The method first evaluates which triangles are safe to remove. The key idea is not "delete a fixed percentage everywhere"; instead, scene-specific risk is inferred from train-view evidence.

What it does:

- Computes train-view surface/render evidence.
- Scores whether a triangle is redundant, unreliable, or risky.
- Allows outdoor scenes to remove around 10-12% of triangles when evidence is stable.
- Protects sensitive indoor scenes with a micro-budget guard, avoiding destructive compression.

Why it matters:

- This produces topology reduction without sacrificing held-out RGB quality.
- It prevents a common failure mode where aggressive deletion improves a compression number but damages geometry or perceptual quality.

### 4.2 Checkpoint-safe topology rewrite

After selecting safe faces, the method rewrites the MeshSplatting checkpoint while preserving checkpoint consistency.

What it does:

- Removes selected faces from the triangle mesh.
- Keeps face/vertex tensor bookkeeping valid.
- Handles face-index remapping.
- Preserves compatibility with the downstream renderer and evaluation scripts.

Why it matters:

- The method remains a usable mesh artifact, not just a post-processing image filter.
- The topology accounting is honest: triangle reduction is measured on the actual compact checkpoint.

### 4.3 Train-only Evidence Lumigraph Adapter

ELA is the main RGB-quality improvement module. It transfers stable residual information from train views to held-out views using rendered RGB, depth, and camera geometry.

Conceptually:

1. Render train views from the current mesh.
2. Compute residuals between train render and train ground truth.
3. For each target view, select nearby support views.
4. Warp support residuals through depth/camera geometry.
5. Aggregate only reliable residual evidence.
6. Apply the correction with train-calibrated alpha and safety guards.

The important safety point is:

> Held-out test images are not used to tune the adapter. The adapter policy is selected from train/calibration evidence.

### 4.4 Guarded adaptive policy

The final Phase-J endpoint adds a guarded policy around ELA:

- Most scenes use adaptive alpha.
- `treehill` uses an auto edge fallback because the edge-aware branch is safer there.
- The policy is evaluated against train-only criteria before being reported on held-out test views.

This is why the method does not reduce to "pick the best test result"; it is a constrained train-only selection policy.

---

## 5. Quantitative Results

### 5.1 Main full9 result vs local clean MeshSplatting

| scene | branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | triangle reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | `24.0215` | `0.7024` | `0.2661` | `+0.7199` | `+0.0425` | `-0.0660` | `11.81%` |
| flowers | adaptive alpha | `20.3044` | `0.5578` | `0.3292` | `+0.6221` | `+0.0459` | `-0.0653` | `11.82%` |
| garden | adaptive alpha | `26.3111` | `0.8278` | `0.1358` | `+1.2819` | `+0.0478` | `-0.0655` | `3.47%` |
| stump | adaptive alpha | `25.5951` | `0.7241` | `0.2639` | `+0.3901` | `+0.0189` | `-0.0301` | `11.82%` |
| treehill | auto edge fallback | `21.2962` | `0.5956` | `0.3363` | `+0.3620` | `+0.0311` | `-0.0697` | `11.81%` |
| room | adaptive alpha | `30.3056` | `0.9057` | `0.1960` | `+1.5584` | `+0.0209` | `-0.0539` | `2.10%` |
| counter | adaptive alpha | `28.4492` | `0.8937` | `0.1865` | `+1.6974` | `+0.0317` | `-0.0655` | `2.10%` |
| kitchen | adaptive alpha | `30.1997` | `0.9161` | `0.1320` | `+2.3812` | `+0.0396` | `-0.0672` | `2.10%` |
| bonsai | adaptive alpha | `31.8620` | `0.9303` | `0.1726` | `+2.9668` | `+0.0339` | `-0.0869` | `11.80%` |

Mean result:

```text
dPSNR  = +1.3311
dSSIM  = +0.0347
dLPIPS = -0.0634
mean triangle reduction = 7.6479%
```

### 5.2 Comparison with MeshSplatting paper table

Using the paper's reported per-scene values and current Phase-J metrics:

| scene | paper PSNR/SSIM/LPIPS | Phase-J PSNR/SSIM/LPIPS | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|
| bicycle | `23.04 / 0.641 / 0.348` | `24.0215 / 0.7024 / 0.2661` | `+0.9815` | `+0.0614` | `-0.0819` |
| flowers | `19.34 / 0.480 / 0.417` | `20.3044 / 0.5578 / 0.3292` | `+0.9644` | `+0.0778` | `-0.0878` |
| garden | `24.70 / 0.762 / 0.217` | `26.3111 / 0.8278 / 0.1358` | `+1.6111` | `+0.0658` | `-0.0812` |
| stump | `24.78 / 0.678 / 0.316` | `25.5951 / 0.7241 / 0.2639` | `+0.8151` | `+0.0461` | `-0.0521` |
| treehill | `20.53 / 0.540 / 0.428` | `21.2962 / 0.5956 / 0.3363` | `+0.7662` | `+0.0556` | `-0.0917` |
| room | `28.52 / 0.873 / 0.271` | `30.3056 / 0.9057 / 0.1960` | `+1.7856` | `+0.0327` | `-0.0750` |
| counter | `26.51 / 0.846 / 0.279` | `28.4492 / 0.8937 / 0.1865` | `+1.9392` | `+0.0477` | `-0.0925` |
| kitchen | `27.42 / 0.858 / 0.227` | `30.1997 / 0.9161 / 0.1320` | `+2.7797` | `+0.0581` | `-0.0950` |
| bonsai | `28.19 / 0.876 / 0.294` | `31.8620 / 0.9303 / 0.1726` | `+3.6720` | `+0.0543` | `-0.1214` |

Mean:

```text
Phase-J minus paper table:
  +1.7017 PSNR
  +0.0555 SSIM
  -0.0865 LPIPS
```

This means the accepted method exceeds both:

- our stronger local clean MeshSplatting baseline;
- the original paper-reported MeshSplatting average.

### 5.3 Per-view and geometry/topology audit

From the current README evidence:

- `244 / 246` held-out views strictly improve PSNR, SSIM, and LPIPS over the selected clean baseline.
- Sparse COLMAP geometry is safe on `9 / 9` scenes.
- Sparse COLMAP geometry is strictly better on `6 / 9` scenes under the max500 audit.
- Mean triangle reduction is `7.6479%`.

This is important for the mentor presentation because the method is not purely an RGB post-processing trick; it also preserves a compact mesh-oriented representation.

---

## 6. Qualitative Evidence

The full-frame difference is often visually subtle because the method improves residual-level errors across many pixels rather than radically changing global layout. The best slides should therefore include both full-frame panels and local error-reduction crops.

Recommended assets:

1. Full-frame fair comparison:
   - [`assets/spcarnet_m360_full9_qualitative_gallery.png`](../../assets/spcarnet_m360_full9_qualitative_gallery.png)

2. Outdoor local detail showcase:
   - [`assets/spcarnet_m360_outdoor_detail_showcase.png`](../../assets/spcarnet_m360_outdoor_detail_showcase.png)

3. Mixed indoor/outdoor "where it helps" showcase:
   - [`assets/spcarnet_m360_where_it_helps_showcase.png`](../../assets/spcarnet_m360_where_it_helps_showcase.png)

4. Historical clean-vs-method montage:
   - [`assets/meshsplatopt_clean_vs_ours_montage.png`](../../assets/meshsplatopt_clean_vs_ours_montage.png)

Recommended talking point:

> Full-frame comparison verifies fairness; local error-reduction crops explain where the method helps perceptually.

Example local crop gains recorded in README:

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | `+0.99 / +0.0616 / -0.0682` | `+2.05` | `24.2%` |
| garden / `00008.png` | `+1.27 / +0.0432 / -0.0551` | `+2.70` | `27.6%` |
| treehill / `00010.png` | `+0.59 / +0.0491 / -0.0881` | `+3.03` | `32.0%` |
| bicycle / `00021.png` | `+1.13 / +0.0385 / -0.0615` | `+1.88` | `17.5%` |
| stump / `00007.png` | `+0.26 / +0.0122 / -0.0208` | `+0.81` | `12.8%` |
| bonsai / `00001.png` | `+2.79 / +0.0063 / -0.0007` | `+3.82` | `43.6%` |

---

## 7. Ablations and Lessons

### 7.1 Clean `26000` vs clean `30000`

Longer training is not automatically better. The current baseline protocol selects clean `26000` over clean `30000` on all nine scenes by held-out score.

This avoids the earlier fairness error of comparing against an arbitrary or weaker clean checkpoint.

### 7.2 Compaction alone is not enough

Topology compaction provides compactness and some geometry benefits, but the headline RGB gains come from the train-only ELA correction. This supports the two-part design:

```text
compact safely first, then repair appearance from train-certified residual evidence
```

### 7.3 ELA policy needs guards

The adapter is powerful but can over-apply residuals. The successful Phase-J endpoint uses:

- train-only alpha selection;
- PSNR / SSIM / LPIPS non-regression checks;
- SSIM-peak guard;
- edge fallback where adaptive alpha is unstable.

### 7.4 Newer Phase-S and v25 attempts are not yet headline improvements

Phase-S and v25 are important research attempts, but they should not be overclaimed.

Current interpretation:

- Phase-S: real representation-level face-local repair infrastructure, but incremental gains over Phase-J remain very small.
- v25: real train-objective change using witness-group CVaR, but medium `bonsai` validation rejected it.
- v26: partially implemented local-trust idea, not yet integrated or validated.

v25 evidence:

| stage | accepted | train-val balanced | report-only test LPIPS | report-only test PSNR | report-only test SSIM |
|---|---:|---:|---:|---:|---:|
| plan | false | `-0.000281` | `+0.0000108` | `+0.000547` | `-0.0000075` |
| candidate-owned refit | false | `-0.000420` | `+0.0001369` | `-0.002649` | `-0.0000739` |
| selector strictfull_s1 | false | `-0.000123` | `+0.0000087` | `+0.0000687` | `-0.0000025` |
| final selected | fallback | `0` | `0` | `0` | `0` |

This is a useful slide because it shows that the project has a strict rejection mechanism and does not promote unsafe tiny gains.

---

## 8. What Is Scientifically Interesting?

The contribution is not merely "we tuned MeshSplatting".

The research idea is a constrained, evidence-certified decision loop:

1. MeshSplatting provides a strong opaque triangle representation.
2. Train-view evidence identifies which topology edits are safe.
3. Train-only residual evidence predicts where held-out appearance can be repaired.
4. Guarded selection prevents test-set leakage and rejects unsafe local corrections.
5. The final artifact stays compact and mesh-compatible.

A concise method name for slides:

> **SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting**

Alternative shorter slide name:

> **MeshSplatOpt Phase-J**

---

## 9. Limitations and Honest Risk

The current method is strong against the clean baseline, but not yet a finished top-conference story by itself.

Key limitations:

1. **ELA is still render-time / adapter-like.** It improves held-out images but is not yet fully distilled into a persistent representation-level model.
2. **Visual improvement can be subtle at full-frame scale.** Local crops show the benefit more clearly than full images.
3. **Newer representation-level branches are weak.** Phase-S/v25/v26 have not yet delivered a large improvement over Phase-J.
4. **Paper-table comparison is secondary.** The primary fair evidence is local clean-baseline reproduction.
5. **Need stronger story around novelty.** The best story is not "we beat MeshSplatting by post-processing"; it is "we introduce a train-only evidence-certified compact-and-repair policy around an opaque mesh representation."

Recommended mentor framing:

> We have a strong accepted endpoint versus MeshSplatting, but the next research step should convert the adapter into a more intrinsic representation-level repair to make the contribution more robust and less like a rendering-side correction.

---

## 10. Suggested PPT Structure

### Slide 1: Title

SPCarNet / MeshSplatOpt: Evidence-Certified Compact Residual Repair for MeshSplatting

### Slide 2: Motivation

MeshSplatting is strong and engine-compatible, but clean checkpoints still have local residual blur/errors and redundant topology.

### Slide 3: Baseline protocol

Show local clean `26000/30000` selection rule:

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

Emphasize: no train-metric baseline selection.

### Slide 4: Method pipeline

Diagram:

```text
Clean MeshSplatting checkpoint
  -> train-view surface evidence
  -> safe triangle compaction
  -> checkpoint-safe topology rewrite
  -> train-only residual evidence
  -> guarded ELA repair
  -> held-out render/eval
```

### Slide 5: Module 1 - safe compaction

Explain CSEF/SOR-style evidence and triangle protection.

### Slide 6: Module 2 - Evidence Lumigraph Adapter

Explain support-view residual warping, alpha calibration, and train-only guard.

### Slide 7: Main full9 table

Use the table in Section 5.1.

### Slide 8: Comparison to MeshSplatting paper

Use the mean table:

```text
paper:       24.78 / 0.728 / 0.310
local clean: 25.15 / 0.749 / 0.288
ours:        26.48 / 0.784 / 0.224
```

### Slide 9: Qualitative results

Use full-frame + local crop:

- `spcarnet_m360_full9_qualitative_gallery.png`
- `spcarnet_m360_outdoor_detail_showcase.png`

### Slide 10: Ablation / why guards matter

Clean `26000` beats `30000`; compaction alone is insufficient; ELA needs SSIM/LPIPS guards.

### Slide 11: Honest negative results

Mention v25 witness-CVaR negative result and v26 incomplete state. This makes the presentation credible.

### Slide 12: Next step

Move from render-time adapter to representation-level local-trust repair / distillation.

---

## 11. Recommended One-Minute Oral Summary

> MeshSplatting is already a strong opaque-mesh renderer, so our goal was not to replace it, but to make it self-diagnose and repair. We add a train-only evidence loop: first, it removes triangles only when train-view surface evidence says this is safe; second, it transfers stable residual appearance information from nearby train views to held-out views through depth/camera-aware evidence; third, it accepts the result only through guarded PSNR/SSIM/LPIPS checks. On our local same-protocol Mip-NeRF360 full9 reproduction, the current Phase-J endpoint beats clean MeshSplatting on all 9 scenes with mean +1.33 PSNR, +0.0347 SSIM, -0.0634 LPIPS, while reducing triangles by 7.65%. It is also above the MeshSplatting paper table average. The honest limitation is that the strongest endpoint still relies on an adapter-like ELA component; newer representation-level repairs are implemented but not yet strong enough, so the next step is to distill this evidence-certified repair into the representation itself.

---

## 12. Artifact Index

Main result:

- [`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`](../../outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md)

Local clean baseline / README:

- [`README.md`](../../README.md)

Qualitative assets:

- [`assets/spcarnet_m360_full9_qualitative_gallery.png`](../../assets/spcarnet_m360_full9_qualitative_gallery.png)
- [`assets/spcarnet_m360_outdoor_detail_showcase.png`](../../assets/spcarnet_m360_outdoor_detail_showcase.png)
- [`assets/spcarnet_m360_where_it_helps_showcase.png`](../../assets/spcarnet_m360_where_it_helps_showcase.png)
- [`assets/meshsplatopt_clean_vs_ours_montage.png`](../../assets/meshsplatopt_clean_vs_ours_montage.png)

Honest latest negative / incomplete branches:

- [`docs/car_model/5-30-WitnessGroupCVaR-v25-Log.md`](5-30-WitnessGroupCVaR-v25-Log.md)
- v26 local-trust code is partially present in [`utils/evidence_lumigraph_adapter.py`](../../utils/evidence_lumigraph_adapter.py), but not yet fully integrated into the PhaseK / selector / autovisual pipeline.

External paper source:

- MeshSplatting arXiv HTML: <https://arxiv.org/html/2512.06818v1>

