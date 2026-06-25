# v104a Subagent Story And Weakness Review

Date: 2026-06-25

Scope: paper-story synthesis and review only. This note reads existing docs and output summaries; it does not introduce new experiments, new metrics, or source changes.

## 0. Short Verdict

The paper-safe current SPCarNet story is still Phase-J/v101, not v104a.

SPCarNet changes clean MeshSplatting from:

```text
train checkpoint -> direct held-out render
```

into:

```text
trained MeshSplatting surface
  -> train-view surface evidence audit
  -> geometry-safe triangle compaction
  -> guarded residual repair
  -> train/policy-val gate and fallback
  -> held-out render evaluation
```

The strongest full9 claim remains Phase-J/v101-style guarded evidence repair over the same local selected-clean MeshSplatting baseline: `9/9` scene strict RGB wins, `244/246` held-out view strict RGB wins, mean `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS, and `7.6479%` mean triangle reduction.

v103/v104a/v104c are a different line: representation-field experiments that try to move the residual repair into a persistent surface-addressed field. v103 is hard-triad positive versus clean but still far below v101/v102a. v104a adds raw view direction to v103 and is hard-triad positive versus both clean and v103. v104c adds centered view-affine fitting with fixed algebraic shrinkage and improves over v104a on every hard-triad scene/metric, but it still trails the v101/v102a endpoint ceiling on all three mean metrics.

## 1. What Changed Versus Clean MeshSplatting

Clean MeshSplatting directly renders the optimized checkpoint. It does not explicitly decide which triangles are safe to remove, which residuals are stable across support views, or when a local correction should fall back to no-op.

SPCarNet adds a post-training evidence layer:

- Surface evidence cache: stores train/support residual, depth, visibility, face/bin support, local risk, and agreement signals.
- Geometry-safe compaction: removes only low-risk triangles under policy evidence, producing the reported triangle reductions.
- Guarded residual repair: transfers train-derived residual evidence through surface correspondence, with local trust and fallback.
- Deployment closure in v101: exposes the Phase-J/v100 endpoint through `render.py` with a forceable evidence bank and detached-package validation.
- Representation-field line in v102b/v103/v104a: attempts to replace render-time support aggregation with a checkpoint-attached field sampled by visible triangle ids.

This should be phrased as "SPCarNet builds on MeshSplatting with evidence-certified post-training compaction and repair," not as a replacement training method or as a vanilla checkpoint that renders without endpoint logic.

## 2. Current State By Layer

| layer | status | paper role |
|---|---|---|
| Phase-J guarded ELA + compaction | full9 headline positive | Main quality/triangle-reduction story. |
| v101 render.py endpoint + evidence bank | full9 deployment closure | Makes Phase-J-style output auditable through `render.py`; not an independent quality gain. |
| v102a preprojected delta bank | hard-triad exact acceleration endpoint | Preserves v101/v102a output on target-camera sets; not a general surface representation. |
| v102b static surface residual field | counter weak/mixed | Negative ablation: one RGB residual per triangle loses most of the v101/v102a signal. |
| v103 face-local affine field | hard-triad positive vs clean | First useful surface-field representation; still much weaker than v101/v102a. |
| v104a view-affine field | hard-triad positive vs clean/v103 | Adds view direction and improves over v103 on counter/kitchen/bonsai; still weaker than v101/v102a. |
| v104c shrink view-affine field | hard-triad positive vs clean/v103/v104a | Centered view-affine fit with fixed shrinkage toward v103 fallback; current best surface-field variant. |

## 3. What v104a Adds Over v103

v103 stores a face-local affine barycentric residual basis:

```text
[1, barycentric_0, barycentric_1]
```

v104a extends that basis with a linear view-direction term:

```text
[1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z]
```

At render time, the endpoint reads visible triangle ids, computes the pixel barycentric basis, estimates triangle-center view direction from the current camera, evaluates the per-triangle RGB coefficients, applies `residual_clip=0.08`, then clamps the final RGB.

Counter result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean counter reference | `26.751774` | `0.862055` | `0.252003` |
| v103 affine min_count=1 | `27.208200` | `0.863405` | `0.243176` |
| v104a view-affine min_count=1 | `27.492378` | `0.867344` | `0.239003` |
| v101/v102a endpoint ceiling | `28.442907` | `0.893696` | `0.186557` |

Counter deltas:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104a minus clean | `+0.740604` | `+0.005288` | `-0.013000` |
| v104a minus v103 | `+0.284178` | `+0.003939` | `-0.004173` |
| v104a minus v101/v102a | `-0.950529` | `-0.026352` | `+0.052446` |

Interpretation: view direction is useful, and v104a closes part of the v103-to-v101 counter gap. It does not close the gap. It is still clearly below v101/v102a on all three counter metrics.

## 4. Hard-Triad Quantitative Evidence

v104a hard-triad evidence is now persisted under:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v104_view_affine_field_hardtriad_20260625/v104a_hardtriad_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v104_view_affine_field_hardtriad_20260625/v104a_hardtriad_summary.md
```

Hard triad is `counter`, `kitchen`, `bonsai`.

Mean hard-triad metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting reference | `27.821853` | `0.878303` | `0.236894` |
| v103 affine min_count=1 | `28.384418` | `0.879855` | `0.226611` |
| v104a view-affine min_count=1 | `28.823045` | `0.884927` | `0.219492` |
| v104c shrink view-affine | `28.859798` | `0.885459` | `0.219064` |
| v101/v102a endpoint ceiling | `30.167395` | `0.913355` | `0.163709` |

Mean deltas:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v103 minus clean | `+0.562565` | `+0.001552` | `-0.010283` |
| v103 minus v101/v102a | `-1.782977` | `-0.033500` | `+0.062902` |
| v104a minus clean | `+1.001192` | `+0.006625` | `-0.017402` |
| v104a minus v103 | `+0.438627` | `+0.005072` | `-0.007120` |
| v104a minus v101/v102a | `-1.344350` | `-0.028428` | `+0.055783` |
| v104c minus clean | `+1.037945` | `+0.007156` | `-0.017830` |
| v104c minus v103 | `+0.475380` | `+0.005604` | `-0.007547` |
| v104c minus v104a | `+0.036753` | `+0.000532` | `-0.000427` |
| v104c minus v101/v102a | `-1.307599` | `-0.027896` | `+0.055355` |

Per-scene v103 deltas versus clean:

| scene | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| counter | `+0.456427` | `+0.001350` | `-0.008827` |
| kitchen | `+0.491600` | `+0.001101` | `-0.004668` |
| bonsai | `+0.739668` | `+0.002207` | `-0.017353` |

Per-scene v103 deltas versus v101/v102a:

| scene | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| counter | `-1.234707` | `-0.030291` | `+0.056619` |
| kitchen | `-1.887243` | `-0.038540` | `+0.062514` |
| bonsai | `-2.226982` | `-0.031669` | `+0.069574` |

This is the important boundary for the v104a/v104c story: view direction is a real representation-level improvement over v103 across the hard triad, and fixed shrinkage gives a further small but consistent gain over raw v104a. However, the gap to v101/v102a remains large. v104c has earned the current hard-triad representation-field claim, but not the final paper headline.

## 5. Qualitative Claim Boundary

Safe qualitative claim:

SPCarNet's Phase-J/v101 evidence shows local held-out error reduction in traceable crops, especially high-frequency texture, residual lighting/detail errors, and indoor object/surface regions where clean MeshSplatting leaves systematic local error. Full-frame differences may be subtle; error maps and crop-level panels are the right qualitative evidence.

Unsafe qualitative claims:

- Do not claim every full-frame image is visibly better by eye.
- Do not claim v104a qualitative improvement beyond the counter smoke unless new panels/results are produced.
- Do not claim v104a has inherited Phase-J/v101 qualitative quality; its counter metrics remain below that ceiling.
- Do not detach crop claims from their manifests and exact scene/view/crop paths.

Recommended wording:

> The robust visual story belongs to Phase-J/v101: surface-bound guarded residual repair reduces local held-out error where train-view evidence is stable. v104a is a representation-field step toward baking that behavior into a view-aware surface field, not yet the visual headline.

## 6. Weaknesses And Reviewer Risks

1. v104c is hard-triad positive but not full9 validated. A paper claim beyond "counter/kitchen/bonsai hard-triad positive" would be premature.

2. v104c still trails v101/v102a. On hard-triad mean it is `-1.307599` PSNR, `-0.027896` SSIM, and `+0.055355` LPIPS relative to the v101/v102a endpoint ceiling.

3. The surface-field line is not yet unseen-camera generalization. v103/v104a fields are distilled from v102 preprojected target-camera deltas and validated on the same target camera set. The fields store no target GT, but this is still not a train-only, unseen-camera residual-field result.

4. Missing v104a stability controls. The current v104a smoke does not yet implement centered per-triangle view features, `triangle_view_counts` or `min_views`, condition/rank diagnostics, scale-aware ridge, fallback to v103 affine coefficients, or reported OOD/fallback fractions.

5. Runtime is a current weakness for the strong endpoint. Existing runtime summaries show Phase-J integrated no-I/O profiling at `951.410896 ms/view`, `1.051071 FPS`, about `27.044247x` slower than compact render-only. v101 detached counter runtime also showed a `1.885235x` wall slowdown versus standard `render.py`. The project should claim quality/compactness/memory where supported, not speed.

6. Phase-J/v101 is not a vanilla baked checkpoint. v101 is a `render.py` endpoint plus evidence bank. v102a preprojects target-camera deltas but is still target-camera caching, not a general persistent representation.

7. Version sprawl can confuse the paper. The clean story should not be a list of v82 through v104a. For paper/PPT: Phase-J/v101 is the supported method story; v102b/v103/v104a are ablations and forward path for representation baking.

## 7. Next Experiments

P0:

- Treat v104c hard-triad as complete and promote it only as a representation-field improvement over v104a/v103/clean, not over v101/v102a.
- The next representation step should address the remaining structural loss from compressing per-pixel guarded residuals into one low-order triangle-local function.
- Add field-specific claim checks: `used_required_field=true`, fail-closed missing-field behavior, target-GT non-use smoke, zero-field/no-view ablation, and exact artifact manifest.

P1:

- Replace target-camera delta distillation with train/policy-val evidence where possible, or clearly label the method as same-camera distillation until that is done.
- Build an ablation ladder in one table: zero field, v102b constant face residual, v103 affine barycentric, v104a view-affine, v104b centered/fallback, and v101/v102a ceiling.
- Produce qualitative counter/hard-triad panels only after v104a or v104b passes hard triad; include error maps and exact crop manifests.

P2:

- Profile any promoted field endpoint against Phase-J/v101 and v102a. The representation-field line only becomes compelling if it keeps meaningful quality while reducing adapter runtime/storage overhead.
- After hard triad, expand to full9 only if the same fixed policy passes the earlier gates without scene-specific tuning.

## 8. Evidence Read

Primary docs and summaries used:

```text
docs/car_model/6-25-SPCarNet-Current-Method-vs-MeshSplatting-Complete-Report.zh.md
docs/car_model/6-25-OfficialProtocol-Refresh-And-PaperLoop-Gap.md
docs/car_model/6-25-v101-Claim-To-Artifact-Manifest.md
docs/car_model/6-25-v101-Subagent-Review-And-PaperStory.md
docs/car_model/6-25-v102-PreprojectedDelta-Acceleration-Log.md
docs/car_model/6-25-v102b-SurfaceResidualField-Prototype-Log.md
docs/car_model/6-25-v103-FaceLocalAffineResidualField-Log.md
docs/car_model/6-25-v103-HardTriad-Min1-Validation-Log.md
docs/car_model/6-25-v104a-ViewAffineCounter-Smoke-Log.md
outputs/carnet/spcarnet/current_evidence_manifest_20260624.md
outputs/carnet/spcarnet/static_rate_profile_20260625/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/rate_distortion_frontier_20260625.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/runtime_adapter_gap_audit.md
```
